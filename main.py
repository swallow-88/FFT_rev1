# ────────────────────────────────────────────────────────────────────────
# 0)   Imports & Android 환경 판별
# ────────────────────────────────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, shutil
import numpy as np
from numpy.fft import fft

from kivy.app    import App
from kivy.clock  import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.uix.popup    import Popup
from kivy.uix.widget   import Widget
from kivy.graphics     import Line, Color
from kivy.utils        import platform
from plyer             import filechooser

# ── Android / Pyjnius ───────────────────────────────────────────────────
ANDROID      = platform == "android"
ANDROID_API  = 0
toast        = None
Uri          = None
OpenableCols = None
activity     = None

try:
    # cast() 가 없는 오래된 pyjnius 환경도 처리
    try:
        from jnius import autoclass, cast
    except Exception:
        from jnius import autoclass
        cast = lambda cls, obj: obj    # 더미 cast
    if ANDROID:
        ANDROID_API  = autoclass("android.os.Build$VERSION").SDK_INT
        PythonAct    = autoclass("org.kivy.android.PythonActivity")
        activity     = PythonAct.mActivity
        Uri          = autoclass("android.net.Uri")
        OpenableCols = autoclass("android.provider.OpenableColumns")
        from plyer import toast
except Exception as e:
    Logger.warning(f"[JNI unavailable] {e}")
    ANDROID = False   # 데스크탑 런 시나리오

# ────────────────────────────────────────────────────────────────────────
# 1)   전역 예외 → crash.log + 팝업  (폰에서 확인 가능)
# ────────────────────────────────────────────────────────────────────────
def _dump_crash(msg: str):
    path = os.path.join(os.getenv("HOME", "/sdcard"), "crash.log")
    with open(path, "a", encoding="utf-8") as fp:
        fp.write("\n"+"="*60+"\n"+datetime.datetime.now().isoformat()+"\n")
        fp.write(msg+"\n")
    Logger.error(msg)

def _exchook(t, v, tb):
    txt = "".join(traceback.format_exception(t, v, tb))
    _dump_crash(txt)
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=txt[:1500]), size_hint=(.9,.9)).open())
sys.excepthook = _exchook

# ────────────────────────────────────────────────────────────────────────
# 2)   SAF content:// URI → 임시파일 복사
#      (Download 폴더 등의 직접 경로는 그대로 통과)
# ────────────────────────────────────────────────────────────────────────
def uri_to_temp(u_str: str) -> str | None:
    if not (ANDROID and u_str and u_str.startswith("content://")):
        return u_str if u_str and os.path.exists(u_str) else None
    try:
        cr  = activity.getContentResolver()
        uri = Uri.parse(u_str)

        # 파일 이름 얻기
        name = "file"
        c = cr.query(uri, [OpenableCols.DISPLAY_NAME], None, None, None)
        if c and c.moveToFirst():
            name = c.getString(0)
        if c: c.close()

        istream  = cr.openInputStream(uri)
        out_path = os.path.join(activity.getCacheDir().getAbsolutePath(),
                                f"{uuid.uuid4().hex}-{name}")

        with open(out_path, "wb") as dst:
            # ── cast() 미지원 디바이스 →  read()/write() 직접 사용
            if hasattr(istream, "read"):
                while True:
                    buf = istream.read(8192)
                    if not buf: break
                    dst.write(buf)
            else:
                bis = cast("java.io.InputStream", istream)
                arr = autoclass("java.nio.ByteBuffer").allocate(8192)
                while True:
                    n = bis.read(arr.array())
                    if n == -1: break
                    dst.write(bytes(arr.array()[:n]))
        istream.close()
        return out_path
    except Exception as e:
        Logger.error(f"URI copy err: {e}")
        return None

# ────────────────────────────────────────────────────────────────────────
# 3)   그래프 위젯
# ────────────────────────────────────────────────────────────────────────
class GraphWidget(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.colors = itertools.cycle([(1,0,0), (0,1,0), (0,0,1)])
        self.pad_x = 80; self.pad_y = 30
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, data_sets, diff_pts, x_max, y_max):
        self.datasets, self.diff = data_sets, diff_pts
        self.max_x, self.max_y = x_max, y_max
        self.redraw()

    # ---------- 내부 ---------- #
    def redraw(self, *_):
        self.canvas.clear()
        if not self.datasets: return
        with self.canvas:
            self._grid(); self._labels()
            col = self.colors
            for pts in self.datasets:
                Color(*next(col)); Line(points=self._scale(pts))
            if self.diff:
                Color(1,1,1); Line(points=self._scale(self.diff))

    def _scale(self, pts):
        w,h = self.width-2*self.pad_x, self.height-2*self.pad_y
        return [c
                for x,y in pts
                for c in (self.pad_x + x/self.max_x*w,
                          self.pad_y + y/self.max_y*h)]

    def _grid(self):
        gx,gy = (self.width-2*self.pad_x)/10, (self.height-2*self.pad_y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.pad_x+i*gx, self.pad_y,
                         self.pad_x+i*gx, self.height-self.pad_y])
            Line(points=[self.pad_x, self.pad_y+i*gy,
                         self.width-self.pad_x, self.pad_y+i*gy])
    def _labels(self):
        for w in list(self.children):
            if isinstance(w, Label): self.remove_widget(w)
        for i in range(11):
            x = self.pad_x+i*(self.width-2*self.pad_x)/10 - 20
            y = self.pad_y-30
            self.add_widget(Label(text=f"{self.max_x/10*i:.1f} Hz",
                                  size_hint=(None,None), size=(60,20), pos=(x,y)))
            y = self.pad_y+i*(self.height-2*self.pad_y)/10 - 10
            self.add_widget(Label(text=f"{self.max_y/10*i:.1e}",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(self.pad_x-70,y)))
            self.add_widget(Label(text=f"{self.max_y/10*i:.1e}",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(self.width-self.pad_x+20,y)))

# ────────────────────────────────────────────────────────────────────────
# 4)   메인 앱
# ────────────────────────────────────────────────────────────────────────
class FFTApp(App):

    # ── 토스트 + 라벨 동시 로그
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        if ANDROID and toast:
            try: toast.toast(msg)
            except: pass

    # ── 저장소 권한
    def _storage_ok(self):
        if not ANDROID: return True
        from android.permissions import check_permission, Permission
        base=[Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        extra=[]
        if ANDROID_API>=33:
            extra=[Permission.READ_MEDIA_IMAGES,
                   Permission.READ_MEDIA_AUDIO,
                   Permission.READ_MEDIA_VIDEO]
        return all(check_permission(p) for p in base+extra)

    def _ask_storage(self):
        if not ANDROID or self._storage_ok(): return
        from android.permissions import request_permissions, Permission
        perms=[Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if ANDROID_API>=33:
            perms+= [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]
        request_permissions(perms)

    # ── UI
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.label = Label(text="Select 2 CSV files", size_hint=(1,.1)); root.add_widget(self.label)
        root.add_widget(Button(text="Select CSV", size_hint=(1,.1), on_press=self.open_chooser))
        self.btn_run = Button(text="FFT RUN", size_hint=(1,.1), disabled=True, on_press=self.run_fft)
        root.add_widget(self.btn_run)
        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))
        self.graph = GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(lambda *_: self._ask_storage(), 0)
        return root

    # ── 파일 선택
    def open_chooser(self,*_):
        if not self._storage_ok():
            self.log("저장소 권한을 먼저 허용하세요"); return
        filechooser.open_file(
            on_selection=self.on_choose,
            multiple=False,
            filters=[("CSV","*.csv")],
            native=True)         # SAF Picker

    def on_choose(self, sel):
        self.log(f"Chooser ⇒ {sel}")
        if not sel or sel == [None]:
            self.btn_run.disabled=True; return
        paths=[]
        for s in sel[:2]:
            p = uri_to_temp(s)
            if not p:
                self.log("파일 복사 실패"); return
            paths.append(p)
        self.paths=paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled=False

    # ── FFT
    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res=[]
        for p in self.paths:
            pts,mx,my = self.csv_fft(p)
            if pts is None:
                self.log("CSV 읽기 오류"); return
            res.append((pts,mx,my))
        if len(res)==1:
            pts,mx,my = res[0]
            Clock.schedule_once(lambda *_: self.graph.update_graph([pts],[],mx,my)); return
        (f1,x1,y1),(f2,x2,y2)=res
        diff=[(f1[i][0], abs(f1[i][1]-f2[i][1])) for i in range(min(len(f1),len(f2)))]
        mx=max(x1,x2); my=max(y1,y2,max(y for _,y in diff))
        Clock.schedule_once(lambda *_: self.graph.update_graph([f1,f2],diff,mx,my))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    # ── CSV → FFT
    @staticmethod
    def csv_fft(path):
        try:
            t,a=[],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a)<2: raise ValueError("too few samples")
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]
            s=np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f,s)), f.max(), s.max()
        except Exception as e:
            Logger.error(f"FFT err: {e}"); return None,0,0


if __name__ == "__main__":
    FFTApp().run()
