"""
FFT Viewer (Android 안정판)
 - CSV 1‧2 개 FFT 비교
 - Realtime Accelerometer FFT (X·Y·Z)
 - (옵션) Mic FFT – 0‧1 500 Hz
"""

# ───────────────── 기본/외부 모듈 ─────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import time, queue
from collections import deque

import numpy as np
from numpy.fft import fft

from kivy.app            import App
from kivy.clock          import Clock
from kivy.logger         import Logger
from kivy.uix.boxlayout  import BoxLayout
from kivy.uix.button     import Button
from kivy.uix.label      import Label
from kivy.uix.widget     import Widget
from kivy.uix.popup      import Popup
from kivy.uix.modalview  import ModalView
from kivy.graphics       import Color, Line
from kivy.utils          import platform
from plyer               import filechooser, accelerometer

# ──────────────── Android 전용 모듈(있으면) ──────────────────────────────
ANDROID = platform == "android"
toast = SharedStorage = Permission = None
check_permission = request_permissions = lambda *_: None
ANDROID_API = 0

if ANDROID:
    try:             from plyer import toast
    except Exception: toast = None
    try:             from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None
    try:
        from android.permissions import check_permission, request_permissions, Permission
    except Exception:
        class _P: READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = \
                  READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = \
                  RECORD_AUDIO = MANAGE_EXTERNAL_STORAGE = ''
        Permission = _P
        check_permission = lambda *_: True
    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception: ANDROID_API = 0

# ──────────────── sounddevice (Mic FFT) 존재 여부 ────────────────────────
try:
    import sounddevice as sd          # p4a recipes 에 빌드해 두었을 때만 OK
    SD_OK = True
except Exception:
    SD_OK = False
    sd = None

# ──────────────── 예외 → /sdcard/fft_crash.log ──────────────────────────
def _dump_crash(msg: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + f"\n{datetime.datetime.now()}\n{msg}\n")
    except Exception:
        pass
    Logger.error(msg)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash", content=Label(text=str(ev)),
                  size_hint=(.9,.9)).open())
sys.excepthook = _ex

# ──────────────── SAF URI → 앱 캐시 파일 ────────────────────────────────
def uri_to_file(uri: str) -> str | None:
    if not uri: return None
    if uri.startswith("file://"):
        path = urllib.parse.unquote(uri[7:]);  return path if os.path.exists(path) else None
    if not uri.startswith("content://"):       return uri  if os.path.exists(uri) else None
    if ANDROID and SharedStorage:
        try: return SharedStorage().copy_from_shared(
                uri, uuid.uuid4().hex, to_downloads=False)
        except Exception as e: Logger.error(f"SAF copy fail: {e}")
    return None

# ──────────────── 그래프 위젯 ────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS = [(1,0,0), (0,1,0), (0,0,1)]
    DIFF_CLR = (1,1,1)
    LINE_W = 2.4

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, ds:list, df:list, xm:float, ym:float):
        self.max_x = max(xm, 1e-6);  self.max_y = max(ym, 1e-6)
        self.datasets = [d for d in (ds or []) if d]
        self.diff     = df or []
        self.redraw()

    # -------- 내부 도우미 ------------------------------------------------
    def _scale(self, pts):
        w,h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts for c in (self.PAD_X+x/self.max_x*w,
                                           self.PAD_Y+y/self.max_y*h)]

    def _grid(self):
        gx,gy=(self.width-2*self.PAD_X)/10,(self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx,self.PAD_Y,
                         self.PAD_X+i*gx,self.height-self.PAD_Y])
            Line(points=[self.PAD_X,self.PAD_Y+i*gy,
                         self.width-self.PAD_X,self.PAD_Y+i*gy])

    def _labels(self):
        # 기존 축‧피크 라벨 제거
        for w in list(self.children):
            if getattr(w,"_axis",False): self.remove_widget(w)

        # X축 간격 결정
        step = 10 if self.max_x<=60 else (100 if self.max_x<=600 else 300)
        n = int(self.max_x//step)+1
        for i in range(n):
            x = self.PAD_X+i*(self.width-2*self.PAD_X)/(n-1)-20
            l = Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                      size=(60,20), pos=(x,self.PAD_Y-28))
            l._axis=True; self.add_widget(l)

        # Y축
        for i in range(11):
            mag=self.max_y*i/10; y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for x in (self.PAD_X-68, self.width-self.PAD_X+8):
                l=Label(text=f"{mag:.1e}", size_hint=(None,None),
                        size=(60,20), pos=(x,y))
                l._axis=True; self.add_widget(l)

    # -------- 메인 그리기 ------------------------------------------------
    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w,"_peak",False): self.remove_widget(w)

        if not self.datasets: return
        peaks=[]

        with self.canvas:
            self._grid(); self._labels()
            for i,pts in enumerate(self.datasets):
                Color(*self.COLORS[i%len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx,fy=max(pts,key=lambda p:p[1]); sx,sy=self._scale([(fx,fy)])[0:2]
                peaks.append((fx,sx,sy))
            if self.diff:
                Color(*self.DIFF_CLR); Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx,sx,sy in peaks:
            l=Label(text=f"▲ {fx:.1f} Hz", size_hint=(None,None),
                    size=(90,22), pos=(sx-30,sy+6)); l._peak=True; self.add_widget(l)
        # Δ 라벨 (CSV 두 개 비교 시)
        if len(peaks)==2:
            delta=abs(peaks[0][0]-peaks[1][0]); bad=delta>1.5
            clr=(1,0,0,1) if bad else (0,1,0,1)
            l=Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                    size_hint=(None,None), size=(200,24),
                    pos=(self.PAD_X, self.height-self.PAD_Y+6),
                    color=clr); l._peak=True; self.add_widget(l)

# ──────────────── 메인 앱 ────────────────────────────────────────────────
class FFTApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        # 가속도
        self.rt_on=False
        self.buf_acc={ax:deque(maxlen=256) for ax in 'xyz'}
        # 마이크
        self.mic_on=False
        self.buf_mic=deque(maxlen=8192)
        self.mic_stream=None

    # ---------- 작은 로그 + 토스트 --------------------------------------
    def log(self,msg:str):
        Logger.info(msg); self.label.text=msg
        if toast: 
            try: toast.toast(msg)
            except Exception: pass

    # ---------- 퍼미션 체크 ---------------------------------------------
    def _ask_perm(self,*_):
        if not ANDROID: self.btn_sel.disabled=False; return
        need=[Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE,
              Permission.RECORD_AUDIO]
        if ANDROID_API>=33:
            need+=[Permission.READ_MEDIA_IMAGES,Permission.READ_MEDIA_AUDIO,Permission.READ_MEDIA_VIDEO]
        def cb(p,g):
            ok=any(g); self.btn_sel.disabled=not ok
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled=False
        else:
            request_permissions(need,cb)

    # ========================  Realtime Accelerometer  ==================
    def toggle_accel(self,*_):
        self.rt_on=not self.rt_on
        self.btn_rt.text=f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel,0)
                threading.Thread(target=self._accel_fft_loop,daemon=True).start()
            except Exception as e:
                self.log(f"Accel err: {e}"); self.toggle_accel()
        else:
            accelerometer.disable()

    def _poll_accel(self,dt):
        if not self.rt_on: return False
        try:
            ax,ay,az=accelerometer.acceleration
            if None in (ax,ay,az): return
            t=time.time()
            for a,v in zip('xyz',(ax,ay,az)):
                self.buf_acc[a].append((t,abs(v)))
        except Exception as e:
            Logger.warning(f"acc read fail {e}")

    def _accel_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.buf_acc[a])<64 for a in 'xyz'): continue
            ds=[]; ymax=0
            for a in 'xyz':
                ts,val=zip(*self.buf_acc[a]); sig=np.asarray(val,float)
                sig-=sig.mean(); sig*=np.hanning(len(sig))
                dt=(ts[-1]-ts[0])/(len(sig)-1)
                f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]
                v=np.abs(fft(sig))[:len(sig)//2]
                m=f<=50; f,v=f[m],v[m]; v=np.convolve(v,np.ones(8)/8,'same')
                ds.append(list(zip(f,v))); ymax=max(ymax,v.max())
            Clock.schedule_once(lambda *_: self.graph.update_graph(ds,[],50,ymax))

    # =============================  Mic  FFT  ===========================
    def toggle_mic(self,*_):
        if not SD_OK:
            self.log("sounddevice 모듈 없음"); return
        self.mic_on=not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self.mic_stream=sd.InputStream(samplerate=44100,channels=1,
                                               dtype='float32',blocksize=512,
                                               callback=self._on_mic_block)
                self.mic_stream.start()
                threading.Thread(target=self._mic_fft_loop,daemon=True).start()
            except Exception as e:
                self.log(f"Mic err: {e}"); self.toggle_mic()
        else:
            if self.mic_stream:
                try:self.mic_stream.stop(); self.mic_stream.close()
                except Exception: pass
                self.mic_stream=None

    def _on_mic_block(self,in_data,frames,info,status):
        if self.mic_on: self.buf_mic.extend(in_data[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.25)
            if len(self.buf_mic)<4096: continue
            sig=np.array(self.buf_mic,dtype=float); self.buf_mic.clear()
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.fftfreq(len(sig),d=1/44100)[:len(sig)//2]
            v=np.abs(fft(sig))[:len(sig)//2]
            m=f<=1500; f,v=f[m],v[m]; v=np.convolve(v,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_: self.graph.update_graph([list(zip(f,v))],[],1500,v.max()))

    # ========================  UI  =====================================
    def build(self):
        root=BoxLayout(orientation='vertical',padding=10,spacing=10)
        self.label=Label(text="Pick 1‧2 CSV / Realtime",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                            on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                            on_press=self.run_fft)
        self.btn_rt=Button(text="Accel FFT (OFF)",size_hint=(1,.1),
                           on_press=self.toggle_accel)
        mic_txt="Mic FFT (OFF)" if SD_OK else "Mic FFT (N/A)"
        self.btn_mic=Button(text=mic_txt,size_hint=(1,.1),
                            on_press=self.toggle_mic, disabled=not SD_OK)
        root.add_widget(self.label); root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run); root.add_widget(self.btn_rt)
        root.add_widget(self.btn_mic)
        root.add_widget(Button(text="EXIT",size_hint=(1,.1),on_press=self.stop))
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

    # ========================  CSV  처리  ===============================
    def open_chooser(self,*_):
        if ANDROID and ANDROID_API>=30:
            try:
                from jnius import autoclass
                if not autoclass("android.os.Environment").isExternalStorageManager():
                    self.log("⚠️ 설정에서 '모든-파일' 권한을 허용해야 합니다"); return
            except Exception: pass
        if ANDROID and SharedStorage:
            try: SharedStorage().open_file(callback=self.on_choose,multiple=True,mime_type="text/*"); return
            except Exception as e: Logger.warning(f"SAF pick fail {e}")
        filechooser.open_file(on_selection=self.on_choose,multiple=True,
                              filters=[("CSV","*.csv")],native=False)

    def on_choose(self,sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real=uri_to_file(raw);  Logger.info(f"{raw} -> {real}")
            if not real: self.log("파일 접근 실패"); return
            paths.append(real)
        self.paths=paths
        self.label.text=" · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled=False

    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._csv_fft_thread,daemon=True).start()

    def _csv_fft_thread(self):
        res=[]
        for p in self.paths:
            ds,xm,ym=self._csv_fft(p)
            if ds is None: self.log("CSV parse err"); return
            res.append((ds,xm,ym))
        if len(res)==1:
            ds,xm,ym=res[0];  Clock.schedule_once(lambda *_: self.graph.update_graph([ds],[],xm,ym))
        else:
            (d1,x1,y1),(d2,x2,y2)=res
            diff=[(d1[i][0],abs(d1[i][1]-d2[i][1])) for i in range(min(len(d1),len(d2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_: self.graph.update_graph([d1,d2],diff,xm,ym))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    @staticmethod
    def _csv_fft(path):
        try:
            t,a=[],[]
            with open(path,encoding='utf-8') as f:
                for r in csv.reader(f):
                    try:t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a)<2: raise ValueError
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]; v=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,v)),50,v.max()
        except Exception as e:
            Logger.error(f"CSV FFT err {e}"); return None,0,0

# ──────────────── 실행 ──────────────────────────────────────────────────
if __name__=="__main__":
    FFTApp().run()
