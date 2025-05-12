"""
FFT CSV viewer  –  SAF( native=True ) 로만 동작
⚠  API 24+ 에서 FileUriExposedException 방지용
"""
import os, csv, uuid, traceback, sys, urllib.parse, numpy as np
from numpy.fft import fft
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.utils        import platform
from plyer             import filechooser

# ─────────────────── crash → /sdcard/fft_crash.log
def _exhook(et,ev,tb):
    txt="".join(traceback.format_exception(et,ev,tb))
    try: open("/sdcard/fft_crash.log","a").write(txt)
    except: pass
    Logger.error(txt)
sys.excepthook=_exhook

ANDROID = platform == "android"
# SAF → cache 복사
def saf_copy(content_uri:str)->str|None:
    if not(ANDROID and content_uri.startswith("content://")):
        # file:// 또는 /storage/…  → 그대로
        if content_uri.startswith("file://"):
            p = urllib.parse.unquote(content_uri[7:])
            return p if os.path.exists(p) else None
        return content_uri if os.path.exists(content_uri) else None
    try:
        from jnius import autoclass, jarray
        act  = autoclass("org.kivy.android.PythonActivity").mActivity
        Uri  = autoclass("android.net.Uri")
        Cols = autoclass("android.provider.OpenableColumns")
        cr   = act.getContentResolver()
        uri  = Uri.parse(content_uri)
        # 이름 얻기
        name="tmp"
        c=cr.query(uri,[Cols.DISPLAY_NAME],None,None,None)
        if c and c.moveToFirst(): name=c.getString(0)
        if c: c.close()
        dst=os.path.join(act.getCacheDir().getAbsolutePath(),
                         f"{uuid.uuid4().hex}-{name}")
        ist=cr.openInputStream(uri)
        buf=jarray('b')(8192)
        with open(dst,"wb") as out:
            while True:
                n=ist.read(buf)
                if n==-1: break
                out.write(bytes(buf[:n]))
        ist.close()
        Logger.info(f"SAF copy → {dst}")
        return dst
    except Exception as e:
        Logger.error(f"SAF copy fail: {e}")
        return None

# ─────────────────── Kivy UI (선택 → FFT만)
class FFTApp(App):
    def build(self):
        box=BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.lbl=Label(text="Pick up to 2 CSV"); box.add_widget(self.lbl)
        box.add_widget(Button(text="Select",
                              on_press=lambda *_: filechooser.open_file(
                                  self._picked,
                                  multiple=True,
                                  filters=[("CSV","*.csv")],
                                  native=True)   # ← SAF 모드!!
                              ))
        self.btn=Button(text="RUN",disabled=True,on_press=self._run)
        box.add_widget(self.btn)
        return box

    def _picked(self, paths):
        Logger.info(f"PICK ⇒ {paths}")
        if not paths: return
        real=[]
        for p in paths[:2]:
            r=saf_copy(p)
            if not r:
                self.lbl.text="copy fail"; return
            real.append(r)
        self.paths=real
        self.lbl.text=" · ".join(os.path.basename(p) for p in real)
        self.btn.disabled=False

    # very short FFT demo
    def _run(self,*_):
        self.btn.disabled=True
        out=[]
        for p in self.paths:
            t,a=[],[]
            with open(p) as f:
                for r in csv.reader(f):
                    try:t.append(float(r[0]));a.append(float(r[1]))
                    except:pass
            if len(a)<2: continue
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            out.append(f"·{os.path.basename(p)}  max={v.max():.2f}")
        self.lbl.text="\n".join(out); self.btn.disabled=False

if __name__=="__main__":
    FFTApp().run()
