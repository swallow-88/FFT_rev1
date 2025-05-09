"""
FFT CSV viewer  –  min‐crash, SAF ready
"""
import os, csv, threading, itertools, uuid, urllib.parse, numpy as np
from numpy.fft import fft
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label  import Label
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from plyer import filechooser
import traceback, sys

# ──────────────────────────────────  crash log
def dump_crash(et,ev,tb):
    txt="".join(traceback.format_exception(et,ev,tb))
    try: open("/sdcard/fft_crash.log","a").write(txt)
    except: pass
    Logger.error(txt)
sys.excepthook=dump_crash

# ──────────────────────────────────  SAF helper
def uri_to_file(path_or_uri:str)->str|None:
    if not path_or_uri: return None
    if path_or_uri.startswith("file://"):
        p=urllib.parse.unquote(path_or_uri[7:])
        return p if os.path.exists(p) else None
    if not path_or_uri.startswith("content://"):
        return path_or_uri if os.path.exists(path_or_uri) else None
    # SAF copy (needs pyjnius)
    try:
        from jnius import autoclass, jarray
        act  = autoclass("org.kivy.android.PythonActivity").mActivity
        Uri  = autoclass("android.net.Uri")
        Cols = autoclass("android.provider.OpenableColumns")
        cr   = act.getContentResolver()
        uri  = Uri.parse(path_or_uri)
        name = "tmp"
        c = cr.query(uri,[Cols.DISPLAY_NAME],None,None,None)
        if c and c.moveToFirst(): name=c.getString(0)
        if c: c.close()
        ist = cr.openInputStream(uri)
        dst = os.path.join(act.getCacheDir().getAbsolutePath(),
                           f"{uuid.uuid4().hex}-{name}")
        buf = jarray('b')(8192)
        with open(dst,"wb") as out:
            while True:
                n=ist.read(buf)
                if n==-1: break
                out.write(bytes(buf[:n]))
        ist.close()
        return dst
    except Exception as e:
        Logger.error(f"SAF copy fail {e}")
        return None

# ──────────────────────────────────  Graph
class Graph(Widget):
    def __init__(s,**kw):
        super().__init__(**kw); s.data=[]
        s.cols=itertools.cycle([(1,0,0),(0,1,0),(0,0,1)])
        s.bind(size=lambda *_: s.redraw())
    def update(s,*sets): s.data=sets; s.redraw()
    def redraw(s):
        s.canvas.clear()
        if not s.data: return
        with s.canvas:
            for pts in s.data:
                Color(*next(s.cols))
                Line(points=[c for x,y in pts for c in (x*10+40,y*100+40)])

# ──────────────────────────────────  App
class FFTApp(App):
    def build(self):
        root=BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.info=Label(text="pick csv"); root.add_widget(self.info)
        root.add_widget(Button(text="Select",on_press=self.pick))
        self.run=Button(text="RUN",disabled=True,on_press=self.do); root.add_widget(self.run)
        self.g=Graph(); root.add_widget(self.g)
        return root
    def pick(self,*_):
        filechooser.open_file(self.got,multiple=True,filters=[("CSV","*.csv")],native=True)
    def got(self,sel):
        Logger.info(f"pick {sel}")
        if not sel: return
        paths=[]
        for p in sel[:2]:
            real=uri_to_file(p)
            if not real: self.info.text="copy fail"; return
            paths.append(real)
        self.paths=paths; self.info.text=" · ".join(os.path.basename(p) for p in paths)
        self.run.disabled=False
    def do(self,*_):
        self.run.disabled=True
        threading.Thread(target=self.fft,daemon=True).start()
    def fft(self):
        out=[]
        for p in self.paths:
            t,a=[],[]
            with open(p) as f:
                for r in csv.reader(f):
                    try:t.append(float(r[0]));a.append(float(r[1]))
                    except:pass
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            out.append(list(zip(f/5,v/max(v))))
        Clock.schedule_once(lambda *_:(self.g.update(*out),setattr(self.run,"disabled",False)))

if __name__=="__main__":
    FFTApp().run()
