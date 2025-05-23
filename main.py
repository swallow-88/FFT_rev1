# -*- coding: utf-8 -*-
"""
FFT CSV / Accel 실시간 뷰어 (안정판 2025-05-XX)
 - CSV  : 0–50 Hz  2 곡선+차이
 - 센서 : 0–50 Hz  X‧Y‧Z 3 곡선
 - Mic  : 안내만 (sounddevice 없음)
"""
# ──────────────────────────────────────────────────────
import os, csv, sys, time, uuid, queue, datetime, traceback
from collections import deque
import numpy as np
from numpy.fft import fft
# Kivy
from kivy.app            import App
from kivy.clock          import Clock
from kivy.logger         import Logger
from kivy.uix.boxlayout  import BoxLayout
from kivy.uix.button     import Button
from kivy.uix.label      import Label
from kivy.uix.widget     import Widget
from kivy.uix.modalview  import ModalView
from kivy.uix.popup      import Popup
from kivy.graphics       import Line, Color
from kivy.utils          import platform
# plyer
from plyer               import filechooser, accelerometer, toast
# Android-전용(있으면)
ANDROID   = platform == "android"
SharedStorage = None
try:
    if ANDROID:
        from androidstorage4kivy import SharedStorage
        from android.permissions import (check_permission,
                                         request_permissions,
                                         Permission)
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    else:
        Permission = None
        check_permission = lambda *_: True
        request_permissions = lambda *_: None
except Exception:
    SharedStorage = None
    check_permission = lambda *_: True
    request_permissions = lambda *_: None
    Permission = None
    ANDROID_API = 0
# ── 공통 util ──────────────────────────────────────────
def uri_to_file(uri:str):
    if not uri: return None
    if uri.startswith("file://"):
        p = uri[7:]; p = os.path.abspath(p)
        return p if os.path.exists(p) else None
    if uri.startswith("content://") and ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(uri, uuid.uuid4().hex,
                                                    to_downloads=False)
        except Exception as e:
            Logger.warning(f"SAF copy fail: {e}")
    return uri if os.path.exists(uri) else None

def write_crash(msg:str):
    try:
        with open("/sdcard/fft_crash.log","a",encoding="utf-8") as f:
            f.write("\n"+"="*60+"\n"+datetime.datetime.now().isoformat()+"\n")
            f.write(msg+"\n")
    except Exception:
        pass
    Logger.error(msg)

def excepthook(et,ev,tb):
    write_crash("".join(traceback.format_exception(et,ev,tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Crash",content=Label(text=str(ev)),
                  size_hint=(.8,.4)).open())
sys.excepthook = excepthook
# ──────────────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X,PAD_Y = 80,30
    COLORS  = [(1,0,0),(0,1,0),(0,0,1)]
    DIFF    = (1,1,1)
    W       = 2.2
    def __init__(self,**kw):
        super().__init__(**kw)
        self.ds=[]; self.diff=[]
        self.mx=self.my=1
        self.bind(size=self.redraw)
    def update(self,ds,diff,xm,ym):
        self.ds = [d for d in ds if d]
        self.diff = diff or []
        self.mx = max(1e-6,float(xm))
        self.my = max(1e-6,float(ym))
        self.redraw()
    def _scale(self,pts):
        w,h=self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts
                  for c in (self.PAD_X+x/self.mx*w,
                            self.PAD_Y+y/self.my*h)]
    def _grid(self):
        gx,gy=(self.width-2*self.PAD_X)/10,(self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx,self.PAD_Y,
                         self.PAD_X+i*gx,self.height-self.PAD_Y])
            Line(points=[self.PAD_X,self.PAD_Y+i*gy,
                         self.width-self.PAD_X,self.PAD_Y+i*gy])
    def _labels(self):
        for w in list(self.children):
            if getattr(w,'_axis',False): self.remove_widget(w)
        # X-axis 간격
        if   self.mx<=60:   step=10
        elif self.mx<=600:  step=100
        else:               step=300
        n=int(self.mx//step)+1
        for i in range(n):
            x=self.PAD_X+i*(self.width-2*self.PAD_X)/(n-1)-20
            lab=Label(text=f"{i*step:d} Hz",size_hint=(None,None),
                      size=(60,20),pos=(x,self.PAD_Y-28)); lab._axis=True
            self.add_widget(lab)
        # Y
        for i in range(11):
            yv=self.my*i/10
            y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for xx in (self.PAD_X-68,self.width-self.PAD_X+10):
                lab=Label(text=f"{yv:.1e}",size_hint=(None,None),
                          size=(60,20),pos=(xx,y)); lab._axis=True
                self.add_widget(lab)
    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w,'_peak',False): self.remove_widget(w)
        if not self.ds: return
        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            for i,pts in enumerate(self.ds):
                Color(*self.COLORS[i%3]); Line(points=self._scale(pts),width=self.W)
                fx,fy=max(pts,key=lambda p:p[1])
                sx,sy=self._scale([(fx,fy)])[0:2]; peaks.append((fx,sx,sy))
            if self.diff:
                Color(*self.DIFF); Line(points=self._scale(self.diff),width=self.W)
        for fx,sx,sy in peaks:
            lb=Label(text=f"▲{fx:.1f} Hz",size_hint=(None,None),
                     size=(80,22),pos=(sx-25,sy+4)); lb._peak=True
            self.add_widget(lb)
# ──────────────────────────────────────────────────────
class FFTApp(App):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.paths=[]
        # realtime
        self.rt_on=False
        self.buf={'x':deque(maxlen=256),'y':deque(maxlen=256),'z':deque(maxlen=256)}
        # mic
        self.mic_on=False
    # ── UI ──
    def build(self):
        root=BoxLayout(orientation='vertical',padding=8,spacing=6)
        self.label=Label(text="select CSV or use realtime",size_hint=(1,.08))
        bt_sel=Button(text="Select CSV",on_press=self.open)
        self.bt_run=Button(text="FFT RUN",disabled=True,on_press=self.run_csv)
        self.bt_rt=Button(text="Accel FFT (OFF)",on_press=self.toggle_rt)
        self.bt_mic=Button(text="Mic FFT (UNSUP)",disabled=True,
                           on_press=lambda *_: self.log("⚠ sounddevice 미포함 빌드"))
        root.add_widget(self.label); root.add_widget(bt_sel)
        root.add_widget(self.bt_run); root.add_widget(self.bt_rt)
        root.add_widget(self.bt_mic)
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        root.add_widget(Button(text="EXIT",on_press=self.stop))
        Clock.schedule_once(self._ask_perm,0)
        return root
    def log(self,msg): self.label.text=msg; Logger.info(msg); toast.toast(msg) if ANDROID else None
    # ── 권한 ──
    def _ask_perm(self,*_):
        if not ANDROID: return
        need=[Permission.RECORD_AUDIO]
        def cb(p,g):
            if not all(g): self.log("🔊 오디오 권한 거부됨")
        request_permissions(need,cb)
    # ── CSV 선택 ──
    def open(self,*_):
        filechooser.open_file(on_selection=self.chosen,multiple=True,
                              filters=[("CSV","*.csv")])
    def chosen(self,sel):
        self.paths=[uri_to_file(u) for u in sel[:2] if uri_to_file(u)]
        if self.paths:
            self.bt_run.disabled=False
            self.label.text=" · ".join(os.path.basename(p) for p in self.paths)
    # ── CSV FFT ──
    def run_csv(self,*_):
        self.bt_run.disabled=True
        threading.Thread(target=self._csv_thread,daemon=True).start()
    def _csv_thread(self):
        res=[]
        for p in self.paths:
            t,a=[],[]
            try:
                with open(p) as f:
                    for r in csv.reader(f):
                        t.append(float(r[0])); a.append(float(r[1]))
                if len(a)<2: raise ValueError
                dt=(t[-1]-t[0])/len(a)
                f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
                v=np.abs(fft(a))[:len(a)//2]
                m=f<=50; f,v=f[m],v[m]
                v=np.convolve(v,np.ones(10)/10,'same')
                res.append((list(zip(f,v)),50,v.max()))
            except Exception as e:
                self.log(f"CSV err: {e}"); return
        if not res: return
        if len(res)==1:
            ds,xm,ym=res[0]
            Clock.schedule_once(lambda *_: self.graph.update([ds],[],xm,ym))
        else:
            (d1,x1,y1),(d2,x2,y2)=res
            diff=[(d1[i][0],abs(d1[i][1]-d2[i][1]))
                  for i in range(min(len(d1),len(d2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(v for _,v in diff))
            Clock.schedule_once(lambda *_: self.graph.update([d1,d2],diff,xm,ym))
        Clock.schedule_once(lambda *_: setattr(self.bt_run,'disabled',False))
    # ── 가속도 ──
    def toggle_rt(self,*_):
        self.rt_on=not self.rt_on
        self.bt_rt.text=f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            accelerometer.enable()
            Clock.schedule_interval(self._poll,0)
            threading.Thread(target=self._rt_loop,daemon=True).start()
        else:
            accelerometer.disable()
    def _poll(self,dt):
        if not self.rt_on: return False
        ax,ay,az=accelerometer.acceleration
        if None in (ax,ay,az): return
        ts=time.time()
        self.buf['x'].append((ts,abs(ax)))
        self.buf['y'].append((ts,abs(ay)))
        self.buf['z'].append((ts,abs(az)))
    def _rt_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.buf[a])<64 for a in 'xyz'): continue
            ds=[]; ymax=0
            for a in 'xyz':
                ts,val=zip(*self.buf[a]); sig=np.asarray(val)
                sig-=sig.mean(); sig*=np.hanning(len(sig))
                dt=(ts[-1]-ts[0])/(len(sig)-1)
                f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]
                v=np.abs(fft(sig))[:len(sig)//2]
                m=f<=50; f,v=f[m],v[m]
                v=np.convolve(v,np.ones(8)/8,'same')
                ds.append(list(zip(f,v))); ymax=max(ymax,v.max())
            Clock.schedule_once(lambda *_: self.graph.update(ds,[],50,ymax))
# ──────────────────────────────────────────────────────
if __name__=="__main__":
    FFTApp().run()
