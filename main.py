"""
FFT Viewer — CSV / Realtime Accel / Mic  (Android 전용 안정판)
"""

# ── 표준, 3rd-party ──────────────────────────────────────────────
import os, csv, sys, time, traceback, threading, datetime, uuid, urllib.parse
from collections import deque

import numpy as np
from numpy.fft import fft

from kivy.app            import App
from kivy.clock          import Clock
from kivy.logger         import Logger
from kivy.uix.boxlayout  import BoxLayout
from kivy.uix.button     import Button
from kivy.uix.label      import Label
from kivy.uix.popup      import Popup
from kivy.uix.widget     import Widget
from kivy.graphics       import Line, Color
from kivy.utils          import platform
from plyer               import filechooser, accelerometer

# ── (선택) 사운드 입력 ----------------------------------------------------
try:
    import sounddevice as sd               # buildozer.spec 에 sounddevice 추가
    HAVE_SD = True
except Exception:
    HAVE_SD = False

# ── Android 전용 모듈 ----------------------------------------------------
ANDROID = platform == "android"
toast  = None
SharedStorage = None
Permission = check_permission = request_permissions = None
ANDROID_API = 0

if ANDROID:
    try: from plyer import toast
    except Exception: pass

    try: from androidstorage4kivy import SharedStorage
    except Exception: pass

    try:
        from android.permissions import (
            Permission, check_permission, request_permissions)
    except Exception:
        class _P: READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
        Permission = _P
        check_permission  = lambda *_: True
        request_permissions = lambda *_: None

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception: pass

# ── 전역 예외 → /sdcard/fft_crash.log -----------------------------------
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" + datetime.datetime.now().isoformat()
                     + "\n" + txt + "\n")
    except Exception:
        pass
    Logger.error(txt)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)), size_hint=(.9,.9)).open())
sys.excepthook = _ex

# ── SAF URI → 캐시 파일 --------------------------------------------------
def uri_to_file(u: str) -> str | None:
    if not u: return None
    if u.startswith("file://"):
        real = urllib.parse.unquote(u[7:])
        return real if os.path.exists(real) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                u, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.warning(f"SAF copy fail: {e}")
    return None

# ── 간단 그래프 위젯 -----------------------------------------------------
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0),(0,1,0),(0,0,1)]   # 빨/초/파
    LINE_W   = 2.5

    def __init__(self, **kw):
        super().__init__(**kw)
        self.data = []          # [[(x,y),…], …]  up to 3 lines
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update(self, datasets:list, max_x:float, max_y:float):
        self.data  = [d for d in datasets if d]
        self.max_x = max(1e-6, max_x)
        self.max_y = max(1e-6, max_y)
        self.redraw()

    # ---------- 내부 도우미 -----------------------------------
    def _scale(self, pts):
        w, h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts
                   for c in (self.PAD_X + x/self.max_x*w,
                             self.PAD_Y + y/self.max_y*h)]

    def _draw_axes(self):
        gx, gy = (self.width-2*self.PAD_X)/10, (self.height-2*self.PAD_Y)/10
        Color(.5,.5,.5);   # grid
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])
        # remove old labels
        for ch in self.children[:]:
            if getattr(ch,"_axis",False) or getattr(ch,"_peak",False):
                self.remove_widget(ch)
        # X-tick 간격
        if   self.max_x <=  60: step = 10
        elif self.max_x <= 600: step = 100
        else:                   step = 300
        nx = int(self.max_x//step)+1
        for i in range(nx):
            x = self.PAD_X+i*(self.width-2*self.PAD_X)/(nx-1)-20
            l = Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                      size=(60,20), pos=(x,self.PAD_Y-28)); l._axis=True
            self.add_widget(l)
        # Y
        for i in range(11):
            mag=self.max_y*i/10
            y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for x in (self.PAD_X-70, self.width-self.PAD_X+10):
                l=Label(text=f"{mag:.1e}",size_hint=(None,None),
                        size=(60,20),pos=(x,y)); l._axis=True
                self.add_widget(l)

    # ---------- 주 그리기 ------------------------------------
    def redraw(self,*_):
        self.canvas.clear()
        if not self.data: return
        with self.canvas:
            self._draw_axes()
            peaks=[]
            for idx,pts in enumerate(self.data):
                Color(*self.COLORS[idx%len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx,fy=max(pts,key=lambda p:p[1])
                sx,sy=self._scale([(fx,fy)])[0:2]
                peaks.append((fx,sx,sy))
            # peak labels
            for fx,sx,sy in peaks:
                l=Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None),size=(90,22),
                        pos=(sx-30,sy+6)); l._peak=True; self.add_widget(l)

# ── 메인 앱 --------------------------------------------------------------
class FFTApp(App):

    def __init__(self, **kw):
        super().__init__(**kw)
        # 실시간 가속도
        self.rt_on=False
        self.acc_buf={axis:deque(maxlen=256) for axis in 'xyz'}
        # 마이크
        self.mic_on=False
        self.mic_buf=deque(maxlen=8192)
        self.mic_stream=None
        # 파일 FFT
        self.paths=[]

    # ---------- 공용 로그 ------------
    def log(self,msg): Logger.info(msg); self.label.text=msg

    # ---------- 권한 -----------------
    def _ask_perm(self,*_):
        if not ANDROID: return
        need=[Permission.RECORD_AUDIO]
        if not all(check_permission(p) for p in need):
            request_permissions(need, lambda *_:None)

    # ---------- UI -------------------
    def build(self):
        root=BoxLayout(orientation='vertical',padding=8,spacing=6)
        self.label=Label(text="Choose CSV or Realtime options",
                         size_hint=(1,.08))
        root.add_widget(self.label)
        # buttons
        self.btn_csv=Button(text="Select CSV",size_hint=(1,.08),
                            on_press=self.open_csv)
        self.btn_run=Button(text="CSV FFT RUN",size_hint=(1,.08),
                            disabled=True,on_press=self.run_csv_fft)
        self.btn_rt =Button(text="Accel FFT (OFF)",size_hint=(1,.08),
                            on_press=self.toggle_accel)
        self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.08),
                            on_press=self.toggle_mic,
                            disabled=not HAVE_SD)
        for b in (self.btn_csv,self.btn_run,self.btn_rt,self.btn_mic):
            root.add_widget(b)
        root.add_widget(Button(text="EXIT",size_hint=(1,.08),
                               on_press=self.stop))
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

    # ---------- CSV 로드 -------------
    def open_csv(self,*_):
        filechooser.open_file(on_selection=self._csv_chosen,
                              multiple=True, filters=[("CSV","*.csv")])
    def _csv_chosen(self,sel):
        if not sel: return
        self.paths=[uri_to_file(p) or p for p in sel[:2]]
        self.btn_run.disabled=False
        self.log("CSV ready")
    def run_csv_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._csv_fft_bg,daemon=True).start()
    def _csv_fft_bg(self):
        data=[]; ymax=0
        for p in self.paths:
            try:
                t,a=zip(*[(float(r[0]),float(r[1]))
                          for r in csv.reader(open(p))])
            except Exception as e:
                self.log(f"CSV read err:{e}"); return
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            mask=f<=50; f,v=f[mask],v[mask]
            v=np.convolve(v,np.ones(10)/10,'same')
            data.append(list(zip(f,v))); ymax=max(ymax,v.max())
        Clock.schedule_once(lambda *_:
            self.graph.update(data,50,ymax))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    # ---------- 가속도 --------------
    def toggle_accel(self,*_):
        self.rt_on=not self.rt_on
        self.btn_rt.text=f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel,0)
                threading.Thread(target=self._acc_fft_loop,
                                 daemon=True).start()
            except Exception as e:
                self.log(f"accel fail:{e}"); self.toggle_accel()
        else:
            accelerometer.disable()
    def _poll_accel(self,dt):
        if not self.rt_on: return False
        try:
            ax,ay,az=accelerometer.acceleration
            if None in (ax,ay,az): return
            now=time.time()
            for v,a in zip((ax,ay,az),'xyz'):
                self.acc_buf[a].append((now,abs(v)))
        except Exception: pass
    def _acc_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.acc_buf[a])<64 for a in 'xyz'): continue
            sets=[]; ymax=0
            for a in 'xyz':
                ts,val=zip(*self.acc_buf[a]); sig=np.asarray(val,float)
                n=len(sig); dt=(ts[-1]-ts[0])/(n-1)
                sig-=sig.mean(); sig*=np.hanning(n)
                f=np.fft.fftfreq(n,d=dt)[:n//2]
                v=np.abs(fft(sig))[:n//2]; m=f<=50
                f,v=f[m],v[m]; v=np.convolve(v,np.ones(8)/8,'same')
                sets.append(list(zip(f,v))); ymax=max(ymax,v.max())
            Clock.schedule_once(lambda *_:
                self.graph.update(sets,50,ymax))

    # ---------- 마이크 --------------
    def toggle_mic(self,*_):
        if not HAVE_SD:
            self.log("sounddevice not available"); return
        self.mic_on=not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self.mic_stream=sd.InputStream(samplerate=44100,channels=1,
                    dtype='float32',blocksize=512,callback=self._on_audio)
                self.mic_stream.start()
                threading.Thread(target=self._mic_fft_loop,
                                 daemon=True).start()
            except Exception as e:
                self.log(f"mic start err:{e}"); self.toggle_mic()
        else:
            try: self.mic_stream.stop(); self.mic_stream.close()
            except Exception: pass
    def _on_audio(self,indata, frames, t, status):
        if self.mic_on: self.mic_buf.extend(indata[:,0])
    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.3)
            if len(self.mic_buf)<2048: continue
            sig=np.array(self.mic_buf,dtype=float); self.mic_buf.clear()
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.fftfreq(len(sig),d=1/44100)[:len(sig)//2]
            v=np.abs(fft(sig))[:len(sig)//2]; m=f<=1500
            f,v=f[m],v[m]; v=np.convolve(v,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update([list(zip(f,v))],1500,v.max()))

# ── run ───────────────────────────────────────────────────────
if __name__=="__main__":
    FFTApp().run()
