"""
FFT CSV viewer – 권한‧파일선택 크래시 방지 버전
"""

# ───── 기본 import ──────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, uuid, urllib.parse
import numpy as np
from numpy.fft import fft

from kivy.app    import App
from kivy.clock  import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.uix.widget   import Widget
from kivy.graphics     import Line, Color
from kivy.utils        import platform
from plyer             import filechooser

# ───── Android / pyjnius 환경 식별 ─────────────────────────────
ANDROID      = platform == "android"
ANDROID_API  = 0
toast = Uri = Cols = activity = None

try:
    from jnius import autoclass, jarray, cast            # pyjnius 존재할 때
    if ANDROID:
        ANDROID_API  = autoclass("android.os.Build$VERSION").SDK_INT
        activity     = autoclass("org.kivy.android.PythonActivity").mActivity
        Uri          = autoclass("android.net.Uri")
        Cols         = autoclass("android.provider.OpenableColumns")
        from plyer import toast
except Exception as e:
    Logger.warning(f"[pyjnius UNAVAILABLE] {e}")
    ANDROID = False                                       # 데스크탑 실행용

# ───── 전역 크래시 → /sdcard/fft_crash.log ─────────────────────
def _ex_hook(et, ev, tb):
    txt = "".join(traceback.format_exception(et, ev, tb))
    try:
        open("/sdcard/fft_crash.log", "a", encoding="utf-8").write(txt)
    except Exception:
        pass
    Logger.error(txt)
sys.excepthook = _ex_hook


# ───── SAF content:// URI → 캐시 파일 복사 ──────────────────────
def uri_to_file(p: str) -> str | None:
    if not (ANDROID and p.startswith("content://")):
        return p if p and os.path.exists(p) else None
    # file:// URI → 실경로
    if p.startswith("file://"):
        real = urllib.parse.unquote(p[7:])
        return real if os.path.exists(real) else None
    # 전통 경로
    if not p.startswith("content://"):
        return p if os.path.exists(p) else None
    # SAF 복사
    if not ANDROID:
        return None
    try:
        cr  = activity.getContentResolver()
        uri = Uri.parse(p)

        name = "tmp"
        c = cr.query(uri, [Cols.DISPLAY_NAME], None, None, None)
        if c and c.moveToFirst():
            name = c.getString(0)
        if c:
            c.close()

        ist = cr.openInputStream(uri)
        dst = os.path.join(activity.getCacheDir().getAbsolutePath(),
                           f"{uuid.uuid4().hex}-{name}")

        buf = jarray('b')(8192)
        with open(dst, "wb") as out:
            while True:
                n = ist.read(buf)
                if n == -1:
                    break
                out.write(bytes(buf[:n]))
        ist.close()
        return dst
    except Exception:
        Logger.exception("SAF copy fail")     # <-- 어떤 Exception 인지 바로 확인
        return None





# ───── 간단 그래프 위젯 (축/격자 생략 – 필요하면 이전 코드에서 대체) ──
class Graph(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.data = []
        self.colcycle = itertools.cycle([(1,0,0), (0,1,0), (0,0,1)])
        self.bind(size=lambda *_: self.redraw())

    def update(self, *datasets):
        self.data = datasets
        self.redraw()

    def redraw(self):
        self.canvas.clear()
        if not self.data:
            return
        with self.canvas:
            for pts in self.data:
                Color(*next(self.colcycle))
                Line(points=[c for x, y in pts
                             for c in (40 + x*10,          # 단순 스케일
                                       40 + y*100)])

# ───── 메인 앱 ─────────────────────────────────────────────────
class FFTApp(App):

    # ── 작은 로그 헬퍼
    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
        if ANDROID and toast:
            try:
                toast.toast(msg)
            except Exception:
                pass

    # ── 저장소 권한 검사
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
        return all(check_permission(p) for p in base + extra)

    # ── 권한 요청 콜백
    def _on_perm(self, perms, grants):
        if all(grants):
            self.btn_select.disabled = False
            self.log("✅ 저장소 권한 허용 – CSV 선택 가능")
        else:
            self.log("❌ 권한 거부됨 – 파일을 열 수 없습니다")

    # ── UI 구성
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)

        self.label = Label(text="Select 2 CSV files", size_hint=(1, .1))
        root.add_widget(self.label)

        self.btn_select = Button(text="Select CSV", size_hint=(1, .1),
                                 disabled=True,
                                 on_press=self.open_chooser)
        root.add_widget(self.btn_select)

        self.btn_run = Button(text="FFT RUN", size_hint=(1, .1),
                              disabled=True, on_press=self.run_fft)
        root.add_widget(self.btn_run)

        root.add_widget(Button(text="EXIT", size_hint=(1, .1),
                               on_press=self.stop))

        self.graph = Graph(size_hint=(1, .6))
        root.add_widget(self.graph)

        # 최초 권한 요청
        Clock.schedule_once(self.ask_perm, 0)
        return root

    def ask_perm(self, *_):
        if self._storage_ok():
            self.btn_select.disabled = False
            return
        from android.permissions import request_permissions, Permission
        perms = [Permission.READ_EXTERNAL_STORAGE,
                 Permission.WRITE_EXTERNAL_STORAGE]
        if ANDROID_API >= 33:
            perms += [Permission.READ_MEDIA_IMAGES,
                      Permission.READ_MEDIA_AUDIO,
                      Permission.READ_MEDIA_VIDEO]
        request_permissions(perms, self._on_perm)

    # ── 파일 선택 (예외 감싼 뒤 실패 시 SAF fallback)
    def open_chooser(self, *_):
        try:
            filechooser.open_file(self.on_choose,
                                  multiple=True,
                                  filters=[("CSV", "*.csv")],
                                  native=False)
        except Exception:
            Logger.exception("open_file failed")     # <-- 반드시 logcat에 찍힘
            self.log("internal chooser 오류 – SAF로 재시도")
            try:
                filechooser.open_file(self.on_choose,
                                      multiple=True,
                                      filters=[("CSV","*.csv")],
                                      native=True)
            except Exception:
                Logger.exception("SAF chooser failed")   # ← 여기까지 오면 chooser 문제
                self.log("파일 선택기를 열 수 없습니다")

    def on_choose(self, sel):
        self.log(f"Choose → {sel}")
        if not sel:
            self.btn_run.disabled = True
            return

        paths = []
        for raw in sel[:2]:
            real = uri_to_file(raw)
            Logger.info(f"copy {raw} → {real}")
            if not real:
                self.log("복사 실패 / 경로 없음"); return
            paths.append(real)

        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # ── FFT
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        results = []
        for fp in self.paths:
            pts = self.csv_fft(fp)
            if pts is None:
                self.log("CSV 읽기 오류"); return
            results.append(pts)

        Clock.schedule_once(lambda *_:
            self.graph.update(*results))
        Clock.schedule_once(lambda *_:
            setattr(self.btn_run, "disabled", False))

    @staticmethod
    def csv_fft(path):
        try:
            t, a = [], []
            with open(path) as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0])); a.append(float(r[1]))
                    except Exception:
                        pass
            if len(a) < 2:
                raise ValueError
            dt   = (t[-1] - t[0]) / len(a)
            f    = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v    = np.abs(fft(a))[:len(a)//2]
            m    = f <= 50
            f, v = f[m], v[m]
            v    = np.convolve(v, np.ones(10)/10, 'same')
            vmax = v.max()
            return [(x, y / vmax) for x, y in zip(f, v)]   # 0~1 정규화
        except Exception as e:
            Logger.error(f"fft err {e}")
            return None


if __name__ == "__main__":
    FFTApp().run()
