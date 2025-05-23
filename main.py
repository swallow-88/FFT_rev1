"""
FFT Viewer (CSV · ACCEL  · MIC)
- SAF 파일선택 & ‘모든-파일’ 권한 대응
- 그래프 : 0-50 Hz(가속도·CSV) / 0-1 500 Hz(마이크) 자동 전환
"""

############################################################
# 1)  표준 / 써드파티
############################################################
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import time
from collections import deque

import numpy as np
from numpy.fft import fft

from kivy.app      import App
from kivy.clock    import Clock
from kivy.logger   import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color
from kivy.utils         import platform
from plyer              import filechooser, accelerometer

# ── 사운드디바이스는 선택사항(레시피 없으면 ImportError) ─────────
try:
    import sounddevice as sd          # p4a 레시피가 있을 때만!
    SOUND_OK = True
except Exception:
    SOUND_OK = False

############################################################
# 2)  Android 전용 모듈
############################################################
ANDROID = platform == "android"

if ANDROID:
    try:    from androidstorage4kivy import SharedStorage
    except: SharedStorage = None
    try:
        from plyer import toast
    except: toast = None
    try:
        from android.permissions import check_permission, request_permissions, Permission
    except:
        # 빌드에 permissions 레시피가 없을 때 더미
        check_permission = lambda *_: True
        request_permissions = lambda *_: None
        class _P: READ_EXTERNAL_STORAGE=WRITE_EXTERNAL_STORAGE=""
        Permission = _P
    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0
else:
    SharedStorage = None
    toast = None
    ANDROID_API = 0

############################################################
# 3)  공통 헬퍼
############################################################
def toast_log(msg):
    Logger.info(msg)
    if toast:   # 안드로이드 Toast
        try: toast.toast(msg)
        except: pass

def saf_to_file(uri:str)->str|None:
    """
    SAF(content://..) → 임시파일 복사, file:// → 실제경로 반환
    데스크톱에서는 그대로 경로 반환
    """
    if not uri: return None
    if uri.startswith("file://"):
        p = urllib.parse.unquote(uri[7:])
        return p if os.path.exists(p) else None
    if not uri.startswith("content://"):
        return uri if os.path.exists(uri) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(uri, uuid.uuid4().hex, False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

############################################################
# 4)  그래프 위젯
############################################################
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS = [(1,0,0), (0,1,0), (0,0,1)]   # 빨‧초‧파
    LINE_W = 2.4

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets=[]; self.max_x=50; self.max_y=1
        self.bind(size=self.redraw)

    def update(self, datasets:list[list[tuple]], xmax, ymax):
        self.datasets = datasets
        self.max_x = max(xmax,1e-3)
        self.max_y = max(ymax,1e-3)
        self.redraw()

    # ── 내부 ----------------------------------------------------------------
    def _scale(self, pts):
        w,h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts
                for c in (self.PAD_X + x/self.max_x*w,
                           self.PAD_Y + y/self.max_y*h)]

    def _labels(self):
        # 기존 레이블 제거
        self.children[:] = [w for w in self.children if not getattr(w,'_axis',False)]
        # X축 스텝
        step = 10 if self.max_x<=60 else (100 if self.max_x<=600 else 300)
        n = int(self.max_x//step)+1
        for i in range(n):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(n-1) - 20
            lbl = Label(text=f"{i*step} Hz",size_hint=(None,None),size=(60,20),
                        pos=(x,self.PAD_Y-28)); lbl._axis=True; self.add_widget(lbl)
        # Y축
        for i in range(11):
            yv = self.max_y*i/10
            y  = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{yv:.1e}",size_hint=(None,None),size=(60,20),
                            pos=(x,y)); lbl._axis=True; self.add_widget(lbl)

    def redraw(self,*_):
        self.canvas.clear()
        if not self.datasets: return
        with self.canvas:
            # 격자
            gx,gy=(self.width-2*self.PAD_X)/10,(self.height-2*self.PAD_Y)/10
            Color(.6,.6,.6)
            for i in range(11):
                Line(points=[self.PAD_X+i*gx,self.PAD_Y,
                             self.PAD_X+i*gx,self.height-self.PAD_Y])
                Line(points=[self.PAD_X,self.PAD_Y+i*gy,
                             self.width-self.PAD_X,self.PAD_Y+i*gy])
            # 축 라벨
            self._labels()
            # 곡선
            for idx,pts in enumerate(self.datasets):
                Color(*self.COLORS[idx%len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)

############################################################
# 5)  메인 앱
############################################################
class FFTApp(App):
    # ------------ 초기화 & 상태 -----------------------------------------
    def __init__(self, **kw):
        super().__init__(**kw)
        self.paths=[]               # CSV 경로들
        # 가속도 버퍼(128Hz 근사 샘플링)
        self.acc_buf = {ax:deque(maxlen=256) for ax in 'xyz'}
        self.acc_on  = False
        # 마이크
        self.mic_buf = deque(maxlen=8192)
        self.mic_on  = False
        self.mic_stream=None

    # ------------ UI -----------------------------------------------------
    def build(self):
        root=BoxLayout(orientation='vertical',padding=10,spacing=10)
        self.info  = Label(text="CSV 두 개를 선택하거나 실시간 모드를 켜세요", size_hint=(1,.08))
        root.add_widget(self.info)

        self.btn_sel=Button(text="Select CSV",size_hint=(1,.08),disabled=True,
                            on_press=self.select_csv)
        self.btn_run=Button(text="FFT RUN",size_hint=(1,.08),disabled=True,
                            on_press=self.run_csv_fft)
        self.btn_acc=Button(text="Accel FFT (OFF)",size_hint=(1,.08),
                            on_press=self.toggle_accel)
        self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.08),disabled=not SOUND_OK,
                            on_press=self.toggle_mic)

        for b in (self.btn_sel,self.btn_run,self.btn_acc,self.btn_mic):
            root.add_widget(b)
        root.add_widget(Button(text="EXIT",size_hint=(1,.08),on_press=self.stop))

        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)

        Clock.schedule_once(self._ask_storage_perm,0)
        return root

    # ------------ 권한 ----------------------------------------------------
    def _ask_storage_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled=False; return
        need=[Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE]
        def cb(p,g): self.btn_sel.disabled=not any(g)
        if all(check_permission(n) for n in need):
            self.btn_sel.disabled=False
        else: request_permissions(need,cb)

    # ------------ CSV ----------------------------------------------------
    def select_csv(self,*_):
        if ANDROID and SharedStorage:
            SharedStorage().open_file(callback=self._csv_chosen,
                                      multiple=True,mime_type="text/*")
            return
        filechooser.open_file(on_selection=self._csv_chosen,multiple=True,
                              filters=[("CSV","*.csv")])

    def _csv_chosen(self,sel:list):
        self.paths=[saf_to_file(u) for u in sel[:2] if saf_to_file(u)]
        if self.paths:
            self.info.text=" · ".join(os.path.basename(p) for p in self.paths)
            self.btn_run.disabled=False

    def run_csv_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._csv_fft_bg,daemon=True).start()

    def _csv_fft_bg(self):
        out=[]; ymax=0
        for p in self.paths:
            try:
                t,a=[],[]
                with open(p) as f:
                    for r in csv.reader(f):
                        t.append(float(r[0])); a.append(float(r[1]))
                dt=(t[-1]-t[0])/len(a)
                f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
                v=np.abs(fft(a))[:len(a)//2]
                m=f<=50; f,v=f[m],v[m]
                v=np.convolve(v,np.ones(10)/10,'same')
                out.append(list(zip(f,v))); ymax=max(ymax,v.max())
            except Exception as e:
                toast_log(f"CSV parse err: {e}"); return
        Clock.schedule_once(lambda *_:
            self.graph.update(out,50,ymax))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,'disabled',False))

    # ------------ ACCEL --------------------------------------------------
    def toggle_accel(self,*_):
        self.acc_on=not self.acc_on
        self.btn_acc.text=f"Accel FFT ({'ON' if self.acc_on else 'OFF'})"
        if self.acc_on:
            accelerometer.enable()
            Clock.schedule_interval(self._poll_accel,0)
            threading.Thread(target=self._acc_fft_loop,daemon=True).start()
        else:
            accelerometer.disable()

    def _poll_accel(self,dt):
        if not self.acc_on: return False
        ax,ay,az=accelerometer.acceleration
        if None in (ax,ay,az): return
        now=time.time()
        for v,axn in zip((ax,ay,az),'xyz'):
            self.acc_buf[axn].append((now,abs(v)))

    def _acc_fft_loop(self):
        while self.acc_on:
            time.sleep(0.5)
            if any(len(self.acc_buf[a])<64 for a in 'xyz'): continue
            sets=[]; ymax=0
            for ax in 'xyz':
                ts,val=zip(*self.acc_buf[ax])
                sig=np.asarray(val); sig-=sig.mean(); sig*=np.hanning(len(sig))
                dt=(ts[-1]-ts[0])/(len(sig)-1)
                f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]
                v=np.abs(fft(sig))[:len(sig)//2]
                m=f<=50; f,v=f[m],v[m]
                v=np.convolve(v,np.ones(8)/8,'same')
                sets.append(list(zip(f,v))); ymax=max(ymax,v.max())
            Clock.schedule_once(lambda *_: self.graph.update(sets,50,ymax))

    # ------------ MIC (선택) --------------------------------------------
    def toggle_mic(self,*_):
        if not SOUND_OK:
            toast_log("sounddevice 모듈이 없습니다"); return
        self.mic_on=not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            self._start_mic()
        else:
            self._stop_mic()

    def _start_mic(self):
        try:
            self.mic_stream=sd.InputStream(samplerate=44100,channels=1,
                                           blocksize=1024,callback=self._on_mic)
            self.mic_stream.start()
            threading.Thread(target=self._mic_fft_loop,daemon=True).start()
        except Exception as e:
            toast_log(f"mic start fail: {e}"); self.mic_on=False
            self.btn_mic.text="Mic FFT (OFF)"

    def _stop_mic(self):
        try: self.mic_stream.stop(); self.mic_stream.close()
        except: pass

    def _on_mic(self,indata,frames,time_info,status):
        if self.mic_on: self.mic_buf.extend(indata[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.3)
            if len(self.mic_buf)<4096: continue
            sig=np.array(self.mic_buf,dtype=float); self.mic_buf.clear()
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.fftfreq(len(sig),1/44100)[:len(sig)//2]
            v=np.abs(fft(sig))[:len(sig)//2]
            m=f<=1500; f,v=f[m],v[m]
            v=np.convolve(v,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update([list(zip(f,v))],1500,v.max()))

############################################################
# 6)  런
############################################################
if __name__=="__main__":
    FFTApp().run()
