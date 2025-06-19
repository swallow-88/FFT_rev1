"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판 (2025-06-18, hot-fix-1)
"""

# ───────────────────────── imports ──────────────────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time
from collections import deque
import numpy as np
from numpy.fft import fft

from plyer import accelerometer, filechooser
from android.storage import app_storage_path               # ★ 내부 경로
from kivy.app   import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color
from kivy.utils         import platform

ANDROID = platform == "android"

# ---------- Android helpers ----------
toast = SharedStorage = Permission = None
check_permission = request_permissions = lambda *a,**k: True
ANDROID_API = 0
if ANDROID:
    try:
        from plyer import toast
        from androidstorage4kivy import SharedStorage
        from android.permissions import (
            check_permission, request_permissions, Permission)
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:   # 빌드/레시피가 없으면 더미
        class _D: pass
        Permission = _D

# ───────────────────── crash logger ────────────────────────
def _dump_crash(txt:str):
    """
    내부 전용 디렉터리(<app>/files)에 crash log 를 써 둔다.
    외부 퍼미션 없어도 100 % 성공하는 위치.
    """
    try:
        path = os.path.join(app_storage_path(), "fft_crash.log")
        with open(path, "a", encoding="utf-8") as fp:
            fp.write("\n"+"="*60+"\n"+datetime.datetime.now().isoformat()+"\n"+txt+"\n")
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

# ───────────────────── SAF helper ───────────────────────────
def uri_to_file(u:str)->str|None:
    if not u: return None
    if u.startswith("file://"):
        p=urllib.parse.unquote(u[7:]);  return p if os.path.exists(p) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(u, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

# ───────────────────── Graph widget ─────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS = [(1,0,0),(0,1,0),(0,0,1)]
    DIFF_CLR=(1,1,1); LINE_W=2.5
    def __init__(self,**kw):
        super().__init__(**kw); self.datasets=[]; self.diff=[]; self.max_x=self.max_y=1
        self.bind(size=self.redraw)
    def update_graph(self,ds,df,xm,ym):
        self.datasets=[d for d in (ds or []) if d]; self.diff=df or []
        self.max_x=max(1e-6,float(xm)); self.max_y=max(1e-6,float(ym)); self.redraw()
    def _scale(self,pts):
        w,h=self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts for c in (self.PAD_X+x/self.max_x*w, self.PAD_Y+y/self.max_y*h)]
    def _grid(self):
        gx,gy=(self.width-2*self.PAD_X)/10,(self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx,self.PAD_Y,self.PAD_X+i*gx,self.height-self.PAD_Y])
            Line(points=[self.PAD_X,self.PAD_Y+i*gy,self.width-self.PAD_X,self.PAD_Y+i*gy])
    def _labels(self):
        for w in list(self.children):
            if getattr(w,"_axis",False): self.remove_widget(w)
        # X
        for i in range(6):
            lab=Label(text=f"{10*i} Hz",size_hint=(None,None),size=(50,20),
                      pos=(self.PAD_X+i*(self.width-2*self.PAD_X)/5-18,self.PAD_Y-28)); lab._axis=True
            self.add_widget(lab)
        # Y
        for i in range(11):
            yval=self.max_y*i/10; y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for xx in (self.PAD_X-65,self.width-self.PAD_X+5):
                lab=Label(text=f"{yval:.1e}",size_hint=(None,None),size=(60,20),pos=(xx,y)); lab._axis=True
                self.add_widget(lab)
    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w,"_peak",False): self.remove_widget(w)
        if not self.datasets: return
        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            for idx,pts in enumerate(self.datasets):
                Color(*self.COLORS[idx%len(self.COLORS)]); Line(points=self._scale(pts),width=self.LINE_W)
                try:
                    fx,fy=max(pts,key=lambda p:p[1]); sx,sy=self._scale([(fx,fy)])[0:2]; peaks.append((fx,sx,sy))
                except ValueError: pass
            if self.diff:
                Color(*self.DIFF_CLR); Line(points=self._scale(self.diff),width=self.LINE_W)
        for fx,sx,sy in peaks:
            lab=Label(text=f"▲ {fx:.1f} Hz",size_hint=(None,None),size=(90,22),pos=(sx-30,sy+6)); lab._peak=True
            self.add_widget(lab)
        if len(peaks)>=2:
            d=abs(peaks[0][0]-peaks[1][0]); bad=d>1.5; c=(1,0,0,1) if bad else (0,1,0,1)
            txt=f"Δ = {d:.2f} Hz → {'고장' if bad else '정상'}"
            lab=Label(text=txt,size_hint=(None,None),size=(190,24),
                      pos=(self.PAD_X,self.height-self.PAD_Y+6),color=c); lab._peak=True
            self.add_widget(lab)

# ───────────────────── Main App ─────────────────────────────
class FFTApp(App):
    RT_WIN=256
    def __init__(self,**kw):
        super().__init__(**kw); self.rt_on=False; self.rt_buf={a:deque(maxlen=self.RT_WIN) for a in 'xyz'}
    # ---------- log ----------
    def log(self,msg): Logger.info(msg); self.label.text=msg; (toast.toast(msg) if toast else None)
    # ---------- permissions ----------
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage: self.btn_sel.disabled=False; return
        need=[getattr(Permission,"READ_EXTERNAL_STORAGE","")]
        need=[p for p in need if p]
        def _cb(p,g): self.btn_sel.disabled=not any(g)
        if all(check_permission(p) for p in need): self.btn_sel.disabled=False
        else: request_permissions(need,_cb)
    # ---------- chooser ----------
    def open_chooser(self,*_):
        if ANDROID and SharedStorage:
            try: SharedStorage().open_file(callback=self.on_choose,multiple=True,mime_type="text/*"); return
            except Exception: pass
        filechooser.open_file(on_selection=self.on_choose,multiple=True,filters=[("CSV","*.csv")],native=False)
    def on_choose(self,sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real=uri_to_file(raw); Logger.info(f"{raw} → {real}")
            if not real: self.log("❌ 파일 복사 실패"); return
            paths.append(real)
        self.paths=paths; self.label.text=" · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled=False
    # ---------- realtime ----------
    def toggle_realtime(self,*_):
        self.rt_on=not self.rt_on; self.btn_rt.text=f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try: accelerometer.enable()
            except Exception as e: self.log(str(e)); self.rt_on=False; return
            Clock.schedule_interval(self._poll,0); threading.Thread(target=self._rt_loop,daemon=True).start()
        else:
            try: accelerometer.disable()
            except Exception: pass
    def _poll(self,dt):
        if not self.rt_on: return False
        try:
            ax,ay,az=accelerometer.acceleration
            if None in (ax,ay,az): return
            now=time.time()
            self.rt_buf['x'].append((now,abs(ax))); self.rt_buf['y'].append((now,abs(ay)))
            self.rt_buf['z'].append((now,abs(az)))
        except Exception as e: Logger.warning(str(e))
    def _rt_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[a])<64 for a in 'xyz'): continue
            ds=[]; ymax=0
            for ax in 'xyz':
                ts,val=zip(*self.rt_buf[ax]); n=len(val)
                sig=np.asarray(val,float); sig-=sig.mean(); sig*=np.hanning(n)
                dt=(ts[-1]-ts[0])/(n-1)
                f=np.fft.fftfreq(n,d=dt)[:n//2]; a=np.abs(fft(sig))[:n//2]; m=f<=50
                f,a=f[m],a[m]; s=np.convolve(a,np.ones(8)/8,'same'); ds.append(list(zip(f,s))); ymax=max(ymax,s.max())
            Clock.schedule_once(lambda *_: self.graph.update_graph(ds,[],50,ymax))
    # ---------- CSV FFT ----------
    def run_fft(self,*_):
        if not getattr(self,"paths",None): self.log("CSV 먼저 선택"); return
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg,daemon=True).start()
    def _fft_bg(self):
        try:
            res=[]
            for p in self.paths:
                pts,xm,ym=self.csv_fft(p)
                if pts is None: self.log("CSV parse err"); return
                res.append((pts,xm,ym))
            if len(res)==1:
                pts,xm,ym=res[0]
                Clock.schedule_once(lambda *_: self.graph.update_graph([pts],[],xm,ym))
            else:
                (f1,x1,y1),(f2,x2,y2)=res
                diff=[(f1[i][0],abs(f1[i][1]-f2[i][1])) for i in range(min(len(f1),len(f2)))]
                xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
                Clock.schedule_once(lambda *_: self.graph.update_graph([f1,f2],diff,xm,ym))
        except Exception as e:
            _dump_crash(traceback.format_exc())
            Clock.schedule_once(lambda *_: self.log(f"FFT thread ERR: {e}"))
        finally:
            Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))
    @staticmethod
    def csv_fft(path):
        try:
            t,a=[],[];                                # 시간, 값
            with open(path,newline="") as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a)<4: raise ValueError("too few samples")
            dt=(t[-1]-t[0])/(len(a)-1)
            if dt<=0: raise ValueError("invalid time column")
            sig=np.asarray(a,float); sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]; v=np.abs(fft(sig))[:len(sig)//2]
            m=(f>=1)&(f<=50)
            f,v=f[m],v[m]
            if len(v)==0: raise ValueError("no band in 1-50 Hz")
            s=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,s)),50,s.max()
        except Exception as e:
            _dump_crash(traceback.format_exc())
            return None,0,0
    # ---------- UI ----------
    def build(self):
        root=BoxLayout(orientation='vertical',padding=10,spacing=10)
        self.label=Label(text="Pick 1 or 2 CSV files",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),on_press=self.run_fft)
        root.add_widget(self.label); root.add_widget(self.btn_sel); root.add_widget(self.btn_run)
        root.add_widget(Button(text="EXIT",size_hint=(1,.1),on_press=self.stop))
        self.btn_rt=Button(text="Realtime FFT (OFF)",size_hint=(1,.1),on_press=self.toggle_realtime)
        root.add_widget(self.btn_rt)
        self.btn_rec=Button(text="Record 10 s FFT",size_hint=(1,.1),on_press=lambda *_:None) # To-do: 녹화 루틴 생략
        root.add_widget(self.btn_rec)
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

# ───────────────────────── run ──────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
