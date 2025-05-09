# ────────────────────────────────────────────────────────────────
# 0) Imports & Android 환경 판별
# ────────────────────────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid
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

# ── Android / Pyjnius (존재-유무·버전 모두 대응) ──────────────────────
ANDROID      = platform == "android"
ANDROID_API  = 0
toast        = None
Uri          = None
OpenableCols = None
activity     = None

try:
    # pyjnius ≥1.7 은 cast 지원, 그 이전은 except 분기로 더미 cast 정의
    try:
        from jnius import autoclass, cast
    except Exception:
        from jnius import autoclass
        cast = lambda cls, obj: obj          # cast 없는 환경용 더미
    if ANDROID:
        ANDROID_API  = autoclass("android.os.Build$VERSION").SDK_INT
        PythonAct    = autoclass("org.kivy.android.PythonActivity")
        activity     = PythonAct.mActivity
        Uri          = autoclass("android.net.Uri")
        OpenableCols = autoclass("android.provider.OpenableColumns")
        from plyer import toast
except Exception as e:
    Logger.warning(f"[JNI unavailable] {e}")
    ANDROID = False          # 데스크탑에서도 실행 가능하도록

# ────────────────────────────────────────────────────────────────
# 1) 전역 예외 → crash.log + 팝업
# ────────────────────────────────────────────────────────────────
def _dump_crash(txt: str):
    path = os.path.join(os.getenv("HOME", "/sdcard"), "crash.log")
    with open(path, "a", encoding="utf-8") as fp:
        fp.write("\n"+"="*60+"\n"+datetime.datetime.now().isoformat()+"\n")
        fp.write(txt+"\n")
    Logger.error(txt)

def _ex_hook(t, v, tb):
    txt = "".join(traceback.format_exception(t, v, tb))
    _dump_crash(txt)
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=txt[:1500]),
                  size_hint=(.9,.9)).open())
sys.excepthook = _ex_hook

# ────────────────────────────────────────────────────────────────
# 2) SAF content:// URI  →  앱 cache 로 복사
#    (file://·직접경로는 그대로 통과)
# ────────────────────────────────────────────────────────────────
def uri_to_temp(u_str: str) -> str | None:
    if not (ANDROID and u_str and u_str.startswith("content://")):
        return u_str if u_str and os.path.exists(u_str) else None
    try:
        cr  = activity.getContentResolver()
        uri = Uri.parse(u_str)

        # 파일 이름
        name = "file"
        c = cr.query(uri, [OpenableCols.DISPLAY_NAME], None, None, None)
        if c and c.moveToFirst():
            name = c.getString(0)
        if c: c.close()

        istream  = cr.openInputStream(uri)
        out_path = os.path.join(activity.getCacheDir().getAbsolutePath(),
                                f"{uuid.uuid4().hex}-{name}")

        # ▶︎ 핵심: 항상 jarray('b') 버퍼로 읽기
        buf = jarray('b')(8192)   # Java byte[] 8192
        with open(out_path, "wb") as dst:
            while True:
                n = istream.read(buf)   # n: 읽은 byte 수, -1=EOF
                if n == -1:
                    break
                dst.write(bytes(buf[:n]))   # Java 배열 → Python bytes
        istream.close()
        return out_path

    except Exception as e:
        Logger.error(f"URI copy err: {e}")   # ← logcat 에서 원인 확인
        return None
# ────────────────────────────────────────────────────────────────
# 3) 그래프 위젯
# ────────────────────────────────────────────────────────────────
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
        self.max_x, self.max_y   = x_max, y_max
        self.redraw()

    # ---------- 내부 ---------- #
    def redraw(self, *_):
        self.canvas.clear()
        if not self.datasets:
            return
        with self.canvas:
            self._grid()
            self._labels()
            col = self.colors
            for pts in self.datasets:
                Color(*next(col)); Line(points=self._scale(pts))
            if self.diff:
                Color(1,1,1);  Line(points=self._scale(self.diff))

    # 점 → 화면좌표
    def _scale(self, pts):
        w,h = self.width-2*self.pad_x, self.height-2*self.pad_y
        return [c
                for x,y in pts
                for c in (self.pad_x + x/self.max_x*w,
                          self.pad_y + y/self.max_y*h)]

    def _grid(self):
        gx, gy = (self.width-2*self.pad_x)/10, (self.height-2*self.pad_y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.pad_x+i*gx, self.pad_y,
                         self.pad_x+i*gx, self.height-self.pad_y])
            Line(points=[self.pad_x, self.pad_y+i*gy,
                         self.width-self.pad_x, self.pad_y+i*gy])

    def _labels(self):
        # 기존 레이블 제거
        for w in list(self.children):
            if isinstance(w, Label): self.remove_widget(w)

        # X축
        for i in range(11):
            freq = self.max_x/10*i
            x = self.pad_x + i*(self.width-2*self.pad_x)/10 - 20
            y = self.pad_y - 30
            self.add_widget(Label(text=f"{freq:.1f} Hz",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(x,y)))
        # Y축(좌/우)
        for i in range(11):
            mag = self.max_y/10*i
            y   = self.pad_y + i*(self.height-2*self.pad_y)/10 - 10
            self.add_widget(Label(text=f"{mag:.1e}",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(self.pad_x-70, y)))
            self.add_widget(Label(text=f"{mag:.1e}",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(self.width-self.pad_x+20, y)))

# ────────────────────────────────────────────────────────────────
# 4) 메인 앱
# ────────────────────────────────────────────────────────────────
class FFTApp(App):

    # ── 라벨 + 토스트 로그
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        if ANDROID and toast:
            try: toast.toast(msg)
            except: pass

    # ── 저장소 권한
    def _storage_ok(self):
        if not ANDROID:
            return True
        from android.permissions import check_permission, Permission
        base = [Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE]
        extra = []
        if ANDROID_API >= 33:
            extra = [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]
        return all(check_permission(p) for p in base+extra)

    def _ask_storage(self):
        if not ANDROID or self._storage_ok():
            return
        from android.permissions import request_permissions, Permission
        perms = [Permission.READ_EXTERNAL_STORAGE,
                 Permission.WRITE_EXTERNAL_STORAGE]
        if ANDROID_API >= 33:
            perms += [Permission.READ_MEDIA_IMAGES,
                      Permission.READ_MEDIA_AUDIO,
                      Permission.READ_MEDIA_VIDEO]
        request_permissions(perms)

    # ── UI
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.label = Label(text="Select 2 CSV files", size_hint=(1,.1))
        root.add_widget(self.label)

        root.add_widget(Button(text="Select CSV", size_hint=(1,.1),
                               on_press=self.open_chooser))

        self.btn_run = Button(text="FFT RUN", size_hint=(1,.1),
                              disabled=True, on_press=self.run_fft)
        root.add_widget(self.btn_run)

        root.add_widget(Button(text="EXIT", size_hint=(1,.1),
                               on_press=self.stop))

        self.graph = GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)

        Clock.schedule_once(lambda *_: self._ask_storage(), 0)
        return root

    # ── 파일 선택
    def open_chooser(self,*_):
        if not self._storage_ok():
            self.log("저장소 권한을 먼저 허용하세요"); return

        # native=False  ➜  전통 FileChooser (경로 바로 반환)
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True,
                              filters=[("CSV","*.csv")],
                              native=False)

    def on_choose(self, sel):
        self.log(f"Chooser ⇒ {sel}")

        if not sel or sel == [None]:
            self.btn_run.disabled = True
            return

        paths = []
        for s in sel[:2]:
            p = uri_to_temp(s)
            Logger.info(f"COPY → {s} → {p}")
            if not p:
                self.log("❌ 파일 복사 실패 – 다시 선택")
                return
            paths.append(p)

        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # ── FFT
    def run_fft(self,*_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res = []
        for p in self.paths:
            pts, mx, my = self.csv_fft(p)
            if pts is None:
                self.log("CSV 읽기 오류"); return
            res.append((pts, mx, my))

        if len(res) == 1:
            pts,mx,my = res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], mx, my))
            return

        (f1,x1,y1), (f2,x2,y2) = res
        diff = [(f1[i][0], abs(f1[i][1]-f2[i][1]))
                for i in range(min(len(f1), len(f2)))]
        mx = max(x1,x2);  my = max(y1,y2, max(y for _,y in diff))
        Clock.schedule_once(lambda *_:
            self.graph.update_graph([f1,f2], diff, mx, my))
        Clock.schedule_once(lambda *_:
            setattr(self.btn_run, "disabled", False))

    # ── CSV → FFT
    @staticmethod
    def csv_fft(path):
        try:
            t, a = [], []
            with open(path) as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a) < 2:
                raise ValueError("too few samples")

            dt   = (t[-1]-t[0]) / len(a)
            freq = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            vals = np.abs(fft(a))[:len(a)//2]

            mask = freq <= 50
            freq, vals = freq[mask], vals[mask]
            smooth = np.convolve(vals, np.ones(10)/10, 'same')

            return list(zip(freq, smooth)), freq.max(), smooth.max()
        except Exception as e:
            Logger.error(f"FFT err: {e}")
            return None,0,0


if __name__ == "__main__":
    FFTApp().run()
