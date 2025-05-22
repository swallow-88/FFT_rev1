"""
FFT CSV Viewer – Android SAF & ‘모든-파일’ 권한 대응 안정판
"""

# ── 표준 / 3rd-party ──────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import numpy as np
from numpy.fft import fft

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
from plyer               import filechooser                 # SAF 실패 시 fallback

# ── Android 전용 모듈(있을 때만) ──────────────────────────────────
ANDROID = platform == "android"

toast = SharedStorage = check_permission = request_permissions = Permission = None
ANDROID_API = 0
if ANDROID:
    try:  from plyer import toast
    except Exception: toast = None

    try:  from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None

    try:
        from android.permissions import (check_permission,
                                         request_permissions,
                                         Permission)
    except Exception:
        # permissions recipe 가 없을 때용 더미
        check_permission  = lambda *a, **k: True
        request_permissions = lambda *a, **k: None
        class _P:                      # noqa: N801
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
        Permission = _P

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ── 전역 예외 핸들러 → /sdcard/fft_crash.log ──────────────────────
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n"
                     + datetime.datetime.now().isoformat() + "\n" + txt + "\n")
    except Exception:
        pass
    Logger.error(txt)

def _ex(t, v, tb):
    _dump_crash("".join(traceback.format_exception(t, v, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(v)), size_hint=(.9,.9)).open())
sys.excepthook = _ex

# ── SAF URI → 실제(캐시) 파일 경로 ─────────────────────────────────
def uri_to_file(uri: str) -> str | None:
    if not uri:
        return None
    if uri.startswith("file://"):
        path = urllib.parse.unquote(uri[7:])
        return path if os.path.exists(path) else None
    if not uri.startswith("content://"):
        return uri if os.path.exists(uri) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                uri, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

# ── 그래프 위젯 ──────────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0)]        # 1번=빨강, 2번=초록
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.5

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets: list[list[tuple[float,float]]] = []
        self.diff:     list[tuple[float,float]] = []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    # ---------- 외부에서 호출 ----------
    def update_graph(self, ds, df, xm, ym):
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
        self.max_x    = max(xm, 1e-6)
        self.max_y    = max(ym, 1e-6)
        self.redraw()

    # ---------- 내부 유틸 ----------
    def _scale(self, pts):
        w, h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x, y in pts
                  for c in (self.PAD_X + x/self.max_x*w,
                            self.PAD_Y + y/self.max_y*h)]

    def _grid(self):
        gx, gy = (self.width-2*self.PAD_X)/10, (self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        # 이전 축 라벨 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축 : 0~50 Hz 10Hz 간격
        for i in range(6):
            hz  = i*10
            x   = self.PAD_X + i*(self.width-2*self.PAD_X)/5 - 18
            lbl = Label(text=f"{hz:d} Hz", size_hint=(None,None),
                        size=(60,20), pos=(x, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)

        # Y축 : 좌/우 지수표기
        for i in range(11):
            mag = self.max_y*i/10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                            size=(60,20), pos=(x,y))
                lbl._axis = True
                self.add_widget(lbl)

    # ---------- 그리기 ----------
    def redraw(self, *_):
        self.canvas.clear()

        # 이전 피크/Δ 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks = []  # [(fx, fy, sx, sy)]

        with self.canvas:
            self._grid()
            self._labels()

            for idx, pts in enumerate(self.datasets):
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)

                fx, fy = max(pts, key=lambda p: p[1])
                sx, sy = self._scale([(fx, fy)])[0:2]
                peaks.append((fx, fy, sx, sy))

            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(90,22),
                        pos=(sx-30, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # Δ 라벨
        if len(peaks) >= 2:
            delta = abs(peaks[0][0] - peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            txt   = f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}"
            lbl   = Label(text=txt, size_hint=(None,None), size=(200,24),
                          pos=(self.PAD_X, self.height-self.PAD_Y+6),
                          color=clr)
            lbl._peak = True
            self.add_widget(lbl)

# ── 메인 앱 ──────────────────────────────────────────────────────
class FFTApp(App):

    # ---------- 간단 로그 ----------
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ---------- 권한 ----------
    def _ask_perm(self, *_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            return

        need = [Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE]
        if ANDROID_API >= 33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]

        MANAGE = getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)
        if MANAGE:
            need.append(MANAGE)

        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need,
                lambda _,g: setattr(self.btn_sel, "disabled", not any(g)))

    # ---------- UI ----------
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.label   = Label(text="Pick 1 or 2 CSV files", size_hint=(1,.1))
        self.btn_sel = Button(text="Select CSV", disabled=True, size_hint=(1,.1),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",   disabled=True, size_hint=(1,.1),
                              on_press=self.run_fft)

        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))

        self.graph = GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ---------- 파일 선택 ----------
    def open_chooser(self, *_):
        # Android 11+ ‘모든-파일’ 안내
        if ANDROID and ANDROID_API >= 30:
            try:
                from jnius import autoclass
                if not autoclass("android.os.Environment").isExternalStorageManager():
                    self._allfiles_dialog(); return
            except Exception:
                pass

        # SAF 우선
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True,
                                          mime_type="text/*")
                return
            except Exception as e:
                self.log(f"SAF 오류: {e}")

        # 경로 기반 chooser
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True,
                              filters=[("CSV","*.csv")],
                              native=False,
                              path="/storage/emulated/0/Download")

    def _allfiles_dialog(self):
        mv  = ModalView(size_hint=(.8,.35))
        box = BoxLayout(orientation='vertical', spacing=10, padding=10)
        box.add_widget(Label(text="⚠️ CSV 파일을 보려면\n'모든 파일' 권한이 필요합니다.",
                             halign="center"))
        box.add_widget(Button(text="설정으로 이동", size_hint=(1,.4),
                              on_press=lambda *_:(mv.dismiss(),self._goto_perm())))
        mv.add_widget(box); mv.open()

    def _goto_perm(self):
        from jnius import autoclass
        Intent, Settings, Uri = (autoclass("android.content.Intent"),
                                 autoclass("android.provider.Settings"),
                                 autoclass("android.net.Uri"))
        act = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))

    # ---------- 선택 결과 ----------
    def on_choose(self, sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real = uri_to_file(raw)
            if not real:
                self.log("❌ 복사 실패"); return
            paths.append(real)

        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # ---------- FFT ----------
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_thread, daemon=True).start()

    def _fft_thread(self):
        results=[]
        for p in self.paths:
            pts,xm,ym = self._csv_fft(p)
            if pts is None:
                self.log("CSV parse err"); return
            results.append((pts,xm,ym))

        if len(results)==1:
            pts,xm,ym = results[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], xm, ym))
        else:
            (f1,x1,y1),(f2,x2,y2) = results
            diff=[(f1[i][0], abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm = max(x1,x2)
            ym = max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2], diff, xm, ym))
        Clock.schedule_once(lambda *_:
            setattr(self.btn_run, "disabled", False))

    # ---------- CSV → FFT ----------
    @staticmethod
    def _csv_fft(path):
        try:
            t,a = [],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a)<2: raise ValueError("too few samples")
            dt  = (t[-1]-t[0]) / len(a)
            f   = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v   = np.abs(fft(a))[:len(a)//2]
            m   = f <= 50
            f,v = f[m], v[m]
            s   = np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f,s)), 50, s.max()
        except Exception as e:
            Logger.error(f"FFT err {e}")
            return None,0,0

# ── 실행 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
