# -*- coding: utf-8 -*-
"""
FFT CSV Viewer + Realtime Accelerometer / Mic FFT
(SAF & Android “모든-파일” 권한 대응 안정판)
"""

# ─── 표준 / 3rd-party ──────────────────────────────────────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time
from collections import deque
import numpy as np
from numpy.fft import fft

from kivy.app           import App
from kivy.clock         import Clock
from kivy.logger        import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.popup     import Popup
from kivy.uix.modalview import ModalView
from kivy.graphics      import Line, Color
from kivy.utils         import platform
from plyer              import filechooser, accelerometer

# sounddevice 는 recipe 유무에 따라 달라지므로 try import
try:
    import sounddevice as sd
except Exception:
    sd = None

# ─── Android 전용 모듈 (있을 때만) ─────────────────────────────────
ANDROID   = platform == "android"
toast     = None
SharedStorage = None
Permission = check_permission = request_permissions = None
ANDROID_API = 0

if ANDROID:
    try:  from plyer import toast
    except Exception: toast = None

    try:  from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None

    try:
        from android.permissions import (
            Permission, check_permission, request_permissions)
    except Exception:
        # recipe 가 없으면 더미
        class _P: READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
        Permission = _P
        check_permission = lambda *_: True
        request_permissions = lambda *_: None

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ─── 공통 유틸 ─────────────────────────────────────────────────────
def uri_to_file(u: str) -> str | None:
    "SAF content:// → 앱 캐시 파일 경로"
    if not u: return None
    if u.startswith("file://"):
        p = urllib.parse.unquote(u[7:]);  return p if os.path.exists(p) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(u, uuid.uuid4().hex,
                                                    to_downloads=False)
        except Exception as e:
            Logger.warning(f"SAF copy fail {e}")
    return None

# ─── 그래프 위젯 ──────────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS  = [(1,0,0), (0,1,0), (0,0,1)]
    LINE_W  = 2.4
    DIFF_CLR= (1,1,1)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, ds, df, xmax, ymax):
        self.datasets = [d for d in (ds or []) if d]
        self.diff     = df or []
        self.max_x    = max(1e-6, float(xmax))
        self.max_y    = max(1e-6, float(ymax))
        self.redraw()

    # --- 내부 도우미 --------------------------------------------------
    def _scale(self, pts):
        w,h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts
                  for c in (self.PAD_X+x/self.max_x*w,
                            self.PAD_Y+y/self.max_y*h)]

    def _grid(self):
        gx,gy = (self.width-2*self.PAD_X)/10, (self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        for w in list(self.children):
            if getattr(w, "_axis", False): self.remove_widget(w)

        # X-축 간격 자동
        step = 10 if self.max_x<=60 else 100 if self.max_x<=600 else 300
        n    = int(self.max_x//step)+1
        for i in range(n):
            x = self.PAD_X+i*(self.width-2*self.PAD_X)/(n-1)-20
            lab=Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                      size=(60,20), pos=(x,self.PAD_Y-28)); lab._axis=True
            self.add_widget(lab)

        # Y-축
        for i in range(11):
            yv = self.max_y*i/10
            y  = self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lab=Label(text=f"{yv:.1e}", size_hint=(None,None),
                          size=(60,20), pos=(x,y)); lab._axis=True
                self.add_widget(lab)

    # --- 메인 그리기 --------------------------------------------------
    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w, "_peak", False): self.remove_widget(w)
        if not self.datasets: return

        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            for i,pts in enumerate(self.datasets):
                Color(*self.COLORS[i%len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx,fy=max(pts,key=lambda p:p[1])
                sx,sy=self._scale([(fx,fy)])[:2]; peaks.append((fx,sx,sy))
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        for fx,sx,sy in peaks:
            lab=Label(text=f"▲{fx:.1f} Hz", size_hint=(None,None),
                      size=(80,22), pos=(sx-28,sy+6)); lab._peak=True
            self.add_widget(lab)

# ─── 메인 앱 ──────────────────────────────────────────────────────
class FFTApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.csv_paths=[]
        self.rt_on=False
        self.rt_buf={ax:deque(maxlen=256) for ax in 'xyz'}
        # Mic
        self.mic_on=False; self.mic_buf=deque(maxlen=44100)
        self.mic_sr=44100; self.mic_strm=None

    # -------- 작은 로그 --------
    def log(self,msg): self.label.text=msg; Logger.info(msg)

    # -------- UI --------
    def build(self):
        root=BoxLayout(orientation='vertical',padding=10,spacing=10)
        self.label=Label(text="Pick CSV or use sensors",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                            on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                            on_press=self.run_csv_fft)
        self.btn_rt =Button(text="Accel FFT (OFF)",size_hint=(1,.1),
                            on_press=self.toggle_accel)
        if sd is None:
            self.btn_mic=Button(text="Mic FFT (N/A)",disabled=True,size_hint=(1,.1))
        else:
            self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.1),
                                on_press=self.toggle_mic)

        for w in (self.label,self.btn_sel,self.btn_run,
                  self.btn_rt,self.btn_mic,
                  Button(text="EXIT",size_hint=(1,.1),on_press=self.stop)):
            root.add_widget(w)

        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_storage_perm,0)
        return root

    # -------- 권한 요청 --------
    def _ask_storage_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled=False; return
        need=[Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if not all(check_permission(p) for p in need):
            request_permissions(need,lambda *_: setattr(self.btn_sel,'disabled',False))
        else:
            self.btn_sel.disabled=False

    # -------- CSV chooser --------
    def open_chooser(self,*_):
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self._on_choose,
                                          multiple=True,mime_type="text/*");return
            except Exception: pass
        filechooser.open_file(on_selection=self._on_choose,multiple=True,
                              filters=[("CSV","*.csv")],native=False)

    def _on_choose(self,sel):
        if not sel:return
        self.csv_paths=[uri_to_file(p) for p in sel[:2] if uri_to_file(p)]
        self.label.text=" · ".join(os.path.basename(p) for p in self.csv_paths)
        self.btn_run.disabled=not self.csv_paths

    # -------- CSV FFT --------
    @staticmethod
    def _csv_fft(path):
        t,a=[],[]
        with open(path) as f:
            for r in csv.reader(f):
                try:t.append(float(r[0]));a.append(float(r[1]))
                except:pass
        if len(a)<2:return None
        dt=(t[-1]-t[0])/len(a)
        f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
        v=np.abs(fft(a))[:len(a)//2]
        m=f<=50; f,v=f[m],v[m]
        s=np.convolve(v,np.ones(10)/10,'same')
        return list(zip(f,s)),50,s.max()

    def run_csv_fft(self,*_):
        threading.Thread(target=self._csv_fft_bg,daemon=True).start()

    def _csv_fft_bg(self):
        res=[]
        for p in self.csv_paths:
            r=self._csv_fft(p)
            if r is None: self.log("CSV parse error"); return
            res.append(r)
        if len(res)==1:
            pts,x,y=res[0]; Clock.schedule_once(
                lambda *_: self.graph.update_graph([pts],[],x,y))
        else:
            (p1,x1,y1),(p2,x2,y2)=res
            diff=[(p1[i][0],abs(p1[i][1]-p2[i][1]))
                  for i in range(min(len(p1),len(p2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(d[1] for d in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([p1,p2],diff,xm,ym))

    # -------- Accelerometer --------
    def toggle_accel(self,*_):
        self.rt_on=not self.rt_on; self.btn_rt.text=f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try: accelerometer.enable()
            except Exception as e: self.log(str(e)); self.toggle_accel(); return
            Clock.schedule_interval(self._poll_accel,0)
            threading.Thread(target=self._accel_fft_loop,daemon=True).start()
        else:
            try: accelerometer.disable()
            except: pass

    def _poll_accel(self,dt):
        if not self.rt_on: return False
        try:
            ax,ay,az=accelerometer.acceleration
            if None in (ax,ay,az): return
            t=time.time()
            self.rt_buf['x'].append((t,abs(ax)))
            self.rt_buf['y'].append((t,abs(ay)))
            self.rt_buf['z'].append((t,abs(az)))
        except: pass

    def _accel_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[a])<64 for a in 'xyz'): continue
            ds=[]; ymax=0
            for a in 'xyz':
                ts,val=zip(*self.rt_buf[a]); sig=np.array(val)
                sig-=sig.mean(); sig*=np.hanning(len(sig))
                dt=(ts[-1]-ts[0])/(len(sig)-1)
                f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]
                v=np.abs(fft(sig))[:len(sig)//2]
                m=f<=50; f,v=f[m],v[m]
                v=np.convolve(v,np.ones(8)/8,'same')
                ds.append(list(zip(f,v))); ymax=max(ymax,v.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(ds,[],50,ymax))

    # -------- Mic FFT (sounddevice) --------
    def toggle_mic(self,*_):
        if sd is None: return
        self.mic_on=not self.mic_on; self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self.mic_strm=sd.InputStream(channels=1,samplerate=self.mic_sr,
                                             dtype='float32',blocksize=1024,
                                             callback=self._on_mic_block)
                self.mic_strm.start()
                threading.Thread(target=self._mic_loop,daemon=True).start()
            except Exception as e:
                self.log(str(e)); self.toggle_mic()
        else:
            try:self.mic_strm.stop(); self.mic_strm.close()
            except: pass

    def _on_mic_block(self,data,fr,ti,st):
        if self.mic_on: self.mic_buf.extend(data[:,0])

    def _mic_loop(self):
        win=self.mic_sr          # 1 s
        while self.mic_on:
            time.sleep(0.1)
            if len(self.mic_buf)<win: continue
            sig=np.array([self.mic_buf.popleft() for _ in range(win)])
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.rfftfreq(len(sig),1/self.mic_sr); v=np.abs(np.fft.rfft(sig))
            m=f<=1500; f,v=f[m],v[m]; v=np.convolve(v,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f,v))],[],1500,v.max()))

# ─── 런 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
