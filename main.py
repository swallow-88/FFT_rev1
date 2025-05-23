"""
FFT CSV / Accel / Mic Viewer  – Android SAF & 권한 대응 안정판
"""

# ──────────────────── 표준 라이브러리 ────────────────────
import os, csv, sys, traceback, threading, itertools, datetime
import urllib.parse, uuid, time
from collections import deque

# ──────────────────── 과학 계열 ──────────────────────────
import numpy as np
from numpy.fft import fft

# ──────────────────── Kivy / Plyer ───────────────────────
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.graphics import Line, Color
from kivy.utils import platform
from plyer import filechooser, accelerometer

# ──────────────────── Android 전용 (있을 때) ─────────────
ANDROID = platform == "android"
toast = None
SharedStorage = Permission = check_permission = request_permissions = None
ANDROID_API = 0
if ANDROID:
    try:  from plyer import toast
    except Exception: toast = None
    try:  from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None
    try:
        from android.permissions import (Permission,
                                          check_permission,
                                          request_permissions)
    except Exception:
        # 빌드에 permissions recipe가 없으면 더미로 채움
        class _P: READ_EXTERNAL_STORAGE=WRITE_EXTERNAL_STORAGE=READ_MEDIA_AUDIO=MANAGE_EXTERNAL_STORAGE=""
        Permission=_P; check_permission=lambda *a,**k:True; request_permissions=lambda *a,**k:None
    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ──────────────────── 마이크 모듈 (선택) ──────────────────
try:
    import sounddevice as sd          # 안드로이드에선 p4a recipe 필요
    HAS_SD = True
except Exception:
    sd = None
    HAS_SD = False                     # 모듈 없으면 마이크 기능 숨김

# ═══════════════ 예외 → /sdcard/fft_crash.log ═════════════
def _dump_crash(txt):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" +
                     datetime.datetime.now().isoformat() + "\n" + txt + "\n")
    except Exception:
        pass
    Logger.error(txt)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)),
                  size_hint=(.9,.9)).open())
sys.excepthook = _ex

# ═══════════════ SAF URI → 캐시 경로 ══════════════════════
def uri_to_file(u:str)->str|None:
    if not u:                               return None
    if u.startswith("file://"):
        p = urllib.parse.unquote(u[7:])
        return p if os.path.exists(p) else None
    if not u.startswith("content://"):       # 일반 경로
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:            # SAF 복사
        try:
            return SharedStorage().copy_from_shared(u, uuid.uuid4().hex,
                                                    to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

# ═══════════════ 그래프 위젯 ══════════════════════════════
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0), (0,0,1)]    # 빨·초·파
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.4

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets=[];  self.diff=[]
        self.max_x=self.max_y=1
        self.bind(size=self.redraw)

    def update_graph(self, ds, df, xm, ym):
        self.max_x=max(1e-6,float(xm)); self.max_y=max(1e-6,float(ym))
        self.datasets=[d for d in (ds or []) if d]
        self.diff=df or []
        self.redraw()

    # ───────── 좌표 변환 ─────────
    def _scale(self,pts):
        w,h=self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts
                  for c in (self.PAD_X+x/self.max_x*w,
                            self.PAD_Y+y/self.max_y*h)]

    # ───────── 눈금/라벨 ─────────
    def _grid(self):
        gx,gy=(self.width-2*self.PAD_X)/10,(self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx,self.PAD_Y,
                         self.PAD_X+i*gx,self.height-self.PAD_Y])
            Line(points=[self.PAD_X,self.PAD_Y+i*gy,
                         self.width-self.PAD_X,self.PAD_Y+i*gy])

    def _labels(self):
        # 기존 축 라벨 제거
        for w in list(self.children):
            if getattr(w,"_axis",False): self.remove_widget(w)

        # X축 자동 간격
        step=10 if self.max_x<=60 else 100 if self.max_x<=600 else 300
        n=int(self.max_x//step)+1
        for i in range(n):
            x=self.PAD_X+i*(self.width-2*self.PAD_X)/(max(1,n-1))-20
            lab=Label(text=f"{i*step:d} Hz",size_hint=(None,None),
                      size=(60,20),pos=(x,self.PAD_Y-28)); lab._axis=True
            self.add_widget(lab)

        # Y축
        for i in range(11):
            mag=self.max_y*i/10
            y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lab=Label(text=f"{mag:.1e}",size_hint=(None,None),
                          size=(60,20),pos=(x,y)); lab._axis=True
                self.add_widget(lab)

    # ───────── 전체 그리기 ─────────
    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w,"_peak",False): self.remove_widget(w)
        if not self.datasets: return

        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            for idx,pts in enumerate(self.datasets):
                Color(*self.COLORS[idx%len(self.COLORS)])
                Line(points=self._scale(pts),width=self.LINE_W)
                try:
                    fx,fy=max(pts,key=lambda p:p[1])
                    sx,sy=self._scale([(fx,fy)])[0:2]
                    peaks.append((fx,fy,sx,sy))
                except ValueError: pass
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff),width=self.LINE_W)

        for fx,_,sx,sy in peaks:
            lab=Label(text=f"▲ {fx:.1f} Hz",size_hint=(None,None),
                      size=(85,22),pos=(sx-28,sy+6)); lab._peak=True
            self.add_widget(lab)

        if len(peaks)>=2:
            delta=abs(peaks[0][0]-peaks[1][0])
            bad=delta>1.5
            clr=(1,0,0,1) if bad else (0,1,0,1)
            info=Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                       size_hint=(None,None),size=(190,24),
                       pos=(self.PAD_X,self.height-self.PAD_Y+6),
                       color=clr); info._peak=True
            self.add_widget(info)

# ═══════════════ 메인 App ══════════════════════════════════
class FFTApp(App):
    # ───── 초기화 ─────
    def __init__(self,**kw):
        super().__init__(**kw)
        self.paths=[]
        # 가속도
        self.rt_on=False
        self.rt_buf={ax:deque(maxlen=256) for ax in ('x','y','z')}
        # 마이크
        self.mic_on=False
        self.mic_buf=deque(maxlen=44100)          # 1 초 44.1 k 샘플
        self.mic_thread=None

    # ───── Util log ─────
    def log(self,msg):
        Logger.info(msg); self.label.text=msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ───── 권한 체크 (파일 I/O) ─────
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled=False; return
        need=[Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE]
        if getattr(Permission,"MANAGE_EXTERNAL_STORAGE",None):
            need.append(Permission.MANAGE_EXTERNAL_STORAGE)
        def _cb(p,g):
            self.btn_sel.disabled=not any(g)
            if not any(g): self.log("저장소 권한 거부됨")
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled=False
        else:
            request_permissions(need,_cb)

    # ═══════ UI 빌드 ═══════
    def build(self):
        root=BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.label=Label(text="Pick 1 or 2 CSV files",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                            on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                            on_press=self.run_fft)
        root.add_widget(self.label); root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        self.btn_rt=Button(text="Realtime FFT (OFF)",size_hint=(1,.1),
                           on_press=self.toggle_realtime)
        root.add_widget(self.btn_rt)

        # 마이크 버튼 (sounddevice 가 있을 때만 활성)
        if HAS_SD:
            self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.1),
                                on_press=self.toggle_mic)
        else:
            self.btn_mic=Button(text="Mic FFT (미지원)",size_hint=(1,.1),
                                disabled=True)
        root.add_widget(self.btn_mic)

        root.add_widget(Button(text="EXIT",size_hint=(1,.1),on_press=self.stop))
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

    # ═══════ 파일 선택 ═══════
    def open_chooser(self,*_):
        if ANDROID and ANDROID_API>=30:
            from jnius import autoclass
            Env=autoclass("android.os.Environment")
            if not Env.isExternalStorageManager():
                self.log("설정→모든파일 권한 필요"); return
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True,mime_type="text/*")
                return
            except Exception as e: self.log(f"SAF fail {e}")
        filechooser.open_file(on_selection=self.on_choose,multiple=True,
                              filters=[("CSV","*.csv")],native=False)

    def on_choose(self,sel):
        if not sel: return
        self.paths=[]
        for raw in sel[:2]:
            real=uri_to_file(raw)
            if not real: self.log("복사 실패"); return
            self.paths.append(real)
        self.label.text=" · ".join(os.path.basename(p) for p in self.paths)
        self.btn_run.disabled=False

    # ═══════ CSV FFT ═══════
    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg,daemon=True).start()

    def _fft_bg(self):
        res=[]
        for p in self.paths:
            pts,xm,ym=self.csv_fft(p)
            if pts is None: self.log("CSV parse err"); return
            res.append((pts,xm,ym))
        if len(res)==1:
            pts,xm,ym=res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts],[],xm,ym))
        else:
            (f1,x1,y1),(f2,x2,y2)=res
            diff=[(f1[i][0],abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2],diff,xm,ym))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    @staticmethod
    def csv_fft(path):
        try:
            t,a=[],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a)<2: raise ValueError
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]
            s=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,s)), 50, s.max()
        except Exception as e:
            Logger.error(f"CSV FFT err {e}"); return None,0,0

    # ═══════ 가속도 실시간 ═══════
    def toggle_realtime(self,*_):
        self.rt_on=not self.rt_on
        self.btn_rt.text=f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:                              # 시작
            accelerometer.enable()
            Clock.schedule_interval(self._poll_accel,0)
            threading.Thread(target=self._rt_fft_loop,daemon=True).start()
        else:                                       # 중지
            try: accelerometer.disable()
            except Exception: pass

    def _poll_accel(self,dt):
        if not self.rt_on: return False
        try:
            ax,ay,az=accelerometer.acceleration
            if None in (ax,ay,az): return
            now=time.time()
            self.rt_buf['x'].append((now,abs(ax)))
            self.rt_buf['y'].append((now,abs(ay)))
            self.rt_buf['z'].append((now,abs(az)))
        except Exception as e:
            Logger.warning(f"acc read fail {e}")

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(1)                           # 1 초 윈도우
            if any(len(self.rt_buf[a])<128 for a in ('x','y','z')):
                continue
            datasets=[]; ymax=0
            for axis in ('x','y','z'):
                ts,val=zip(*self.rt_buf[axis])
                sig=np.asarray(val,float); n=len(sig)
                dt=(ts[-1]-ts[0])/(n-1)
                sig-=sig.mean(); sig*=np.hanning(n)
                freq=np.fft.fftfreq(n,d=dt)[:n//2]
                amp =np.abs(fft(sig))[:n//2]
                m=freq<=50; freq,amp=freq[m],amp[m]
                sm=np.convolve(amp,np.ones(8)/8,'same')
                datasets.append(list(zip(freq,sm)))
                ymax=max(ymax,sm.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(datasets,[],50,ymax))

    # ═══════ 마이크 실시간 (sounddevice 있을 때) ═══════
    def toggle_mic(self,*_):
        if not HAS_SD: return
        self.mic_on=not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self.stream=sd.InputStream(samplerate=44100,channels=1,
                                           blocksize=2048,dtype='float32',
                                           callback=self._on_mic_block)
                self.stream.start()
                self.mic_thread=threading.Thread(target=self._mic_fft_loop,
                                                 daemon=True); self.mic_thread.start()
            except Exception as e:
                self.log(f"Mic start fail {e}"); self.mic_on=False
                self.btn_mic.text="Mic FFT (OFF)"
        else:
            try: self.stream.stop(); self.stream.close()
            except Exception: pass

    def _on_mic_block(self,in_data,frames,_,__):
        if self.mic_on: self.mic_buf.extend(in_data[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(1)                   # 1 초마다
            if len(self.mic_buf)<44100: continue
            sig=np.array([self.mic_buf.popleft() for _ in range(44100)])
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            freq=np.fft.fftfreq(len(sig),d=1/44100)[:len(sig)//2]
            amp =np.abs(fft(sig))[:len(sig)//2]
            m=freq<=1500; freq,amp=freq[m],amp[m]
            sm=np.convolve(amp,np.ones(16)/16,'same')
            ymax=sm.max()
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(freq,sm))],[],1500,ymax))

# ═══════════════ 실행 ═════════════════════════════════════
if __name__=="__main__":
    FFTApp().run()
