"""
FFT CSV viewer – min-crash patched
"""
import os, csv, threading, itertools, uuid, urllib.parse, traceback, sys
import numpy as np
from numpy.fft import fft

from kivy.app   import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.uix.widget   import Widget
from kivy.graphics     import Line, Color
from kivy.utils        import platform
from plyer             import filechooser

# ───────────────────────── crash log (변경 없음)
def dump_crash(et, ev, tb):
    txt = "".join(traceback.format_exception(et, ev, tb))
    try:
        open("/sdcard/fft_crash.log", "a").write(txt)
    except Exception:
        pass
    Logger.error(txt)
sys.excepthook = dump_crash

# ───────────────────────── Android helper & SAF
ANDROID = platform == "android"
if ANDROID:
    from jnius import autoclass, jarray, JavaException
    ACTIVITY   = autoclass("org.kivy.android.PythonActivity").mActivity
    Uri        = autoclass("android.net.Uri")
    Cols       = autoclass("android.provider.OpenableColumns")
    try:
        from plyer import toast
    except ImportError:
        toast = None
else:
    ACTIVITY = toast = None   # desktop fallback

def uri_to_file(uri: str) -> str | None:
    """content:// → cache 로 복사, file:// → 실제경로, 전통경로는 그대로"""
    if not uri:
        return None
    if uri.startswith("file://"):
        p = urllib.parse.unquote(uri[7:])
        return p if os.path.exists(p) else None
    if not (ANDROID and uri.startswith("content://")):
        return uri if os.path.exists(uri) else None

    # SAF  복사
    try:
        cr = ACTIVITY.getContentResolver()
        u  = Uri.parse(uri)
        name = "tmp"
        c = cr.query(u, [Cols.DISPLAY_NAME], None, None, None)
        if c and c.moveToFirst():
            name = c.getString(0)
        if c:
            c.close()

        dst_path = os.path.join(
            ACTIVITY.getCacheDir().getAbsolutePath(),
            f"{uuid.uuid4().hex}-{name}"
        )

        buf = jarray('b')(8192)
        ist = cr.openInputStream(u)
        with open(dst_path, "wb") as dst:
            while True:
                n = ist.read(buf)
                if n == -1:
                    break
                dst.write(bytes(buf[:n]))
        ist.close()
        return dst_path
    except Exception as e:
        Logger.error(f"SAF copy fail: {e}")
        return None

# ───────────────────────── Graph widget (스케일링 보강)
class Graph(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.data = []
        self.cols = itertools.cycle([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        self.bind(size=lambda *_: self.redraw())

    def update(self, *sets):
        self.data = sets
        self.redraw()

    def redraw(self):
        self.canvas.clear()
        if not self.data:
            return

        # x / y 최대값 계산
        x_max = max(pt[0] for s in self.data for pt in s) or 1
        y_max = max(pt[1] for s in self.data for pt in s) or 1
        pad   = 40
        w     = max(self.width  - 2*pad, 1)
        h     = max(self.height - 2*pad, 1)

        with self.canvas:
            for pts in self.data:
                Color(*next(self.cols))
                # 스케일링
                pl = []
                for x, y in pts:
                    pl.extend((pad + x/x_max * w,
                               pad + y/y_max * h))
                Line(points=pl)

# ───────────────────────── 권한 체크 (Android 6+)
def ensure_storage_permission():
    if not ANDROID:
        return True
    from android.permissions import Permission, check_permission, request_permissions
    perms = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
    if all(check_permission(p) for p in perms):
        return True

    # 비동기로 권한 요청 → 결과는 callback 으로 오므로 여기서는 False 반환
    def cb(_p, _g): ...
    request_permissions(perms, cb)
    return False

# ─────────────────────────  App
class FFTApp(App):
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.info = Label(text="pick csv"); root.add_widget(self.info)

        root.add_widget(Button(text="Select", on_press=self.pick))

        self.run = Button(text="RUN", disabled=True, on_press=self.do)
        root.add_widget(self.run)

        self.g = Graph(); root.add_widget(self.g)
        return root

    # ---------- 파일 선택 ----------
    def pick(self, *_):
        if not ensure_storage_permission():
            self.log("📂 권한을 먼저 허용해 주세요"); return
        try:
            # native=False  → OS 기본 ‘내 파일’ 을 직접 띄워 줌
            filechooser.open_file(self.got,
                                  multiple=True,
                                  filters=[("CSV", "*.csv")],
                                  native=False)
        except JavaException as e:
            Logger.error(f"FileChooser crash: {e}")
            self.log("파일 탐색기를 열 수 없습니다")

    def got(self, sel):
        Logger.info(f"pick {sel}")
        if not sel:
            return
        paths = []
        for p in sel[:2]:
            real = uri_to_file(p)
            if not real:
                self.log("❌ 파일 복사 실패"); return
            paths.append(real)

        self.paths = paths
        self.info.text = " · ".join(os.path.basename(p) for p in paths)
        self.run.disabled = False

    # ---------- FFT ----------
    def do(self, *_):
        self.run.disabled = True
        threading.Thread(target=self.fft, daemon=True).start()

    def fft(self):
        sets = []
        for p in self.paths:
            t, a = [], []
            with open(p) as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0])); a.append(float(r[1]))
                    except Exception:
                        pass
            if len(a) < 2:
                self.log(f"{os.path.basename(p)}: 데이터 부족"); return
            dt   = (t[-1] - t[0]) / len(a)
            freq = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            vals = np.abs(fft(a))[:len(a)//2]
            sets.append(list(zip(freq, vals/vals.max())))

        Clock.schedule_once(lambda *_: (
            self.g.update(*sets),
            setattr(self.run, "disabled", False)
        ))

    # ---------- helper ----------
    def log(self, msg):
        Logger.info(msg)
        self.info.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass


if __name__ == "__main__":
    FFTApp().run()
