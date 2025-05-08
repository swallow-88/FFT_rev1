# main.py  ───────────────────────────────────────────────────────────────
# 2025-05 버전 - Kivy/Plyer FFT 뷰어 (Android 6 ~ 14 호환)

import os, csv, sys, traceback, threading, itertools

import numpy as np
from numpy.fft import fft

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Line, Color
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.utils import platform
from plyer import filechooser, toast

# ── Android util ────────────────────────────────────────────────────────
ANDROID = platform == "android"
if ANDROID:
    from jnius import autoclass
    ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
else:
    ANDROID_API = 0

# ── 전역 예외 훅: 치명 오류 → 팝업 + logcat ───────────────────────────────
def _exc_handler(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    Logger.error(msg)
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Error",
                  content=Label(text=msg[:1500]),
                  size_hint=(.9,.9)).open())
sys.excepthook = _exc_handler


# ── 그래프 위젯 ──────────────────────────────────────────────────────────
class GraphWidget(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.colors = itertools.cycle([(1,0,0), (0,1,0), (0,0,1)])
        self.pad_x = 80; self.pad_y = 30
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    # public API
    def update_graph(self, data_sets, diff_pts, x_max, y_max):
        self.datasets, self.diff = data_sets, diff_pts
        self.max_x, self.max_y = x_max, y_max
        self.redraw()

    # ── internal ────────────────────────────────────────────────────────
    def redraw(self, *_):
        self.canvas.clear()
        if not self.datasets:
            return
        with self.canvas:
            self._draw_grid()
            self._draw_labels()
            # ① 데이터 그래프
            colcycle = self.colors
            for pts in self.datasets:
                Color(*next(colcycle)); Line(points=self._scale_pts(pts))
            # ② 차이 그래프
            if self.diff:
                Color(1,1,1); Line(points=self._scale_pts(self.diff))

    def _scale_pts(self, pts):
        w, h = self.width - 2*self.pad_x, self.height - 2*self.pad_y
        return [coord
                for x, y in pts
                for coord in (self.pad_x + x/self.max_x * w,
                              self.pad_y + y/self.max_y * h)]

    # 그리드 + 축 레이블
    def _draw_grid(self):
        gx = (self.width-2*self.pad_x)/10
        gy = (self.height-2*self.pad_y)/10
        Color(.6,.6,.6)
        for i in range(11):
            # 세로
            Line(points=[self.pad_x+i*gx, self.pad_y,
                         self.pad_x+i*gx, self.height-self.pad_y])
            # 가로
            Line(points=[self.pad_x, self.pad_y+i*gy,
                         self.width-self.pad_x, self.pad_y+i*gy])

    def _draw_labels(self):
        # 모든 기존 Label 제거
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
        # Y축(왼쪽)
        for i in range(11):
            mag = self.max_y/10*i
            y = self.pad_y + i*(self.height-2*self.pad_y)/10 - 10
            x = self.pad_x - 70
            self.add_widget(Label(text=f"{mag:.1e}",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(x,y)))
        # Y축(오른쪽) – diff
        for i in range(11):
            mag = self.max_y/10*i
            y = self.pad_y + i*(self.height-2*self.pad_y)/10 - 10
            x = self.width-self.pad_x + 20
            self.add_widget(Label(text=f"{mag:.1e}",
                                  size_hint=(None,None), size=(60,20),
                                  pos=(x,y)))


# ── 메인 앱 ───────────────────────────────────────────────────────────────
class FFTApp(App):

    # 작은 로그 출력 helper
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        Clock.schedule_once(lambda *_: setattr(self.label, "text", ""), 3)

    # ── 권한 체크 & 요청 ──
    def _storage_perms_ok(self):
        from android.permissions import check_permission, Permission
        base = [Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE]
        extra = []
        if ANDROID_API >= 33:
            extra = [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]
        return all(check_permission(p) for p in base+extra)

    def _ask_storage_perms(self):
        if self._storage_perms_ok():
            return
        from android.permissions import request_permissions, Permission
        perms = [Permission.READ_EXTERNAL_STORAGE,
                 Permission.WRITE_EXTERNAL_STORAGE]
        if ANDROID_API >= 33:
            perms += [Permission.READ_MEDIA_IMAGES,
                      Permission.READ_MEDIA_AUDIO,
                      Permission.READ_MEDIA_VIDEO]
        request_permissions(perms)

    # ── UI 구성 ──
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.label = Label(text="Select 2 CSV files", size_hint=(1,.1))
        root.add_widget(self.label)

        root.add_widget(Button(text="Select CSV", size_hint=(1,.1),
                               on_press=self.open_chooser))

        self.btn_run = Button(text="FFT RUN", disabled=True, size_hint=(1,.1),
                              on_press=self.run_fft)
        root.add_widget(self.btn_run)

        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))

        self.graph = GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)

        # build 끝난 뒤 첫 프레임에서 권한 요청
        if ANDROID:
            Clock.schedule_once(lambda *_: self._ask_storage_perms(), 0)

        return root

    # ── FileChooser ──
    def open_chooser(self, *_):
        if ANDROID and not self._storage_perms_ok():
            self.log("먼저 저장소 권한을 허용해 주세요.")
            return

        filechooser.open_file(on_selection=self.on_chosen,
                              multiple=True,
                              filters=[("CSV", "*.csv")])

    def on_chosen(self, paths):
        self.log(f"Chooser result → {paths}")
        if not paths:
            toast("선택된 파일이 없습니다.")
            return

        self.paths = paths[:2]
        self.btn_run.disabled = False
        self.label.text = " · ".join(os.path.basename(p) for p in self.paths)

    # ── FFT 처리 ──
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res = []
        for fp in self.paths:
            pts, xm, ym = self.csv_to_fft(fp)
            if pts is None:
                self.log(f"{os.path.basename(fp)} 처리 실패")
                return
            res.append((pts, xm, ym))

        if len(res) == 1:
            f1,x1,y1 = res[0]
            Clock.schedule_once(lambda *_: self.graph.update_graph([f1], [], x1, y1))
            return

        (f1,x1,y1),(f2,x2,y2) = res
        diff = [(f1[i][0], abs(f1[i][1]-f2[i][1])) for i in range(min(len(f1),len(f2)))]
        mx = max(x1,x2); my = max(y1,y2,max(y for _,y in diff))
        Clock.schedule_once(lambda *_: self.graph.update_graph([f1,f2], diff, mx, my))
        Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))

    # ── CSV → FFT ──
    @staticmethod
    def csv_to_fft(path):
        try:
            t, a = [], []
            with open(path) as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a) < 2:
                raise ValueError("not enough samples")

            dt = (t[-1]-t[0]) / len(a)
            freq = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            vals = np.abs(fft(a))[:len(a)//2]
            mask = freq <= 50
            freq, vals = freq[mask], vals[mask]
            smooth = np.convolve(vals, np.ones(10)/10, mode='same')
            return list(zip(freq, smooth)), freq.max(), smooth.max()
        except Exception as e:
            Logger.error(f"FFT error: {e}")
            return None,0,0


if __name__ == "__main__":
    FFTApp().run()
