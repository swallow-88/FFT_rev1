"""
FFT Viewer for Android   2025-05 stable snapshot
– CSV 파일 비교 + 실시간 가속도 + (선택) 마이크 FFT
"""

# ── 표준 / 3rd-party ──────────────────────────────────────────────
import os, csv, sys, time, math, traceback, threading, datetime, uuid, urllib.parse
from collections import deque
import numpy as np
from numpy.fft import fft
from kivy.app            import App
from kivy.clock          import Clock
from kivy.logger         import Logger
from kivy.uix.boxlayout  import BoxLayout
from kivy.uix.button     import Button
from kivy.uix.label      import Label
from kivy.uix.popup      import Popup
from kivy.uix.widget     import Widget
from kivy.uix.modalview  import ModalView
from kivy.graphics       import Line, Color
from kivy.utils          import platform
from plyer               import filechooser, accelerometer          # accelerometer 는 없으면 No-op 객체 반환
# sounddevice 는 선택 사항 – 없으면 마이크 FFT 비활성
try:
    import sounddevice as sd
except Exception:
    sd = None

# ── Android 전용 플래그 / 모듈 ───────────────────────────────────
ANDROID          = platform == "android"
toast            = None
SharedStorage    = None
Permission       = None
check_permission = request_permissions = None
ANDROID_API      = 0

if ANDROID:
    try: from plyer import toast
    except Exception: pass
    try: from androidstorage4kivy import SharedStorage
    except Exception: pass
    try:
        from android.permissions import (
            check_permission, request_permissions, Permission)
    except Exception:
        # permissions recipe 가 없으면 더미
        check_permission  = lambda *_: True
        request_permissions = lambda *_1, **_2: None
        class _P:
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
            MANAGE_EXTERNAL_STORAGE = RECORD_AUDIO = ""
        Permission = _P
    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        pass

# ── 예외 로그 → /sdcard/fft_crash.log ────────────────────────────
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" + datetime.datetime.now().isoformat()
                     + "\n" + txt + "\n")
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

# ── SAF URI → 로컬 캐시 경로 변환 ────────────────────────────────
def uri_to_file(uri: str) -> str | None:
    if not uri:
        return None
    if uri.startswith("file://"):
        p = urllib.parse.unquote(uri[7:])
        return p if os.path.exists(p) else None
    if not uri.startswith("content://"):
        return uri if os.path.exists(uri) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(uri, uuid.uuid4().hex,
                                                    to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None


# ── 그래프 위젯 ──────────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0), (0,0,1)]           # 빨,초,파 (더 오면 순환)
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.2

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets: list[list[tuple[float,float]]] = []
        self.diff    : list[tuple[float,float]]       = []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    # 외부에서 호출
    def update_graph(self, ds, df, xm, ym):
        self.datasets = [d for d in (ds or []) if d]
        self.diff     = df or []
        self.max_x    = max(1e-6, float(xm))
        self.max_y    = max(1e-6, float(ym))
        self.redraw()

    # ── 내부 도우미 ────────────────────────────────────────────
    def _scale(self, pts):
        w, h = self.width - 2*self.PAD_X, self.height - 2*self.PAD_Y
        return [coord
                for x, y in pts
                for coord in (self.PAD_X + x/w * (w*self.max_x)/self.max_x,
                              self.PAD_Y + y/h * (h*self.max_y)/self.max_y)]

    def _grid(self):
        gx, gy = (self.width-2*self.PAD_X)/10, (self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        # 낡은 축/피크 라벨 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X 축 – max_x 기준 간격
        if   self.max_x <=  60: step = 10
        elif self.max_x <= 600: step = 100
        else:                   step = 300      # 0-1500 Hz
        n = int(self.max_x//step)+1
        for i in range(max(2,n)):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(max(1,n-1)) - 20
            lbl = Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                        size=(60,20), pos=(x, self.PAD_Y-30))
            lbl._axis = True
            self.add_widget(lbl)

        # Y 축 지수
        for i in range(11):
            mag = self.max_y * i/10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                            size=(60,20), pos=(x,y))
                lbl._axis = True
                self.add_widget(lbl)

    # ── 메인 그리기 ────────────────────────────────────────────
    def redraw(self, *_):
        self.canvas.clear()
        # 이전 피크 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks = []

        with self.canvas:
            self._grid()
            self._labels()

            # 데이터 곡선
            for idx, pts in enumerate(self.datasets):
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx, fy = max(pts, key=lambda p: p[1])
                sx, sy = self._scale([(fx,fy)])[0:2]
                peaks.append((fx, sx, sy))

            # 차이선
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(90,22),
                        pos=(sx-30, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # 두 개 일 때 Δ 표시
        if len(peaks) >= 2:
            delta = abs(peaks[0][0]-peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            info  = Label(text=f"Δ={delta:.2f} Hz → {'고장' if bad else '정상'}",
                          size_hint=(None,None), size=(200,24),
                          pos=(self.PAD_X, self.height-self.PAD_Y+6),
                          color=clr)
            info._peak = True
            self.add_widget(info)


# ── 메인 앱 ────────────────────────────────────────────────────
class FFTApp(App):

    # 초기화
    def __init__(self, **kw):
        super().__init__(**kw)
        # 실시간 가속도 상태
        self.rt_on   = False
        self.rt_buf  = {ax: deque(maxlen=256) for ax in ('x','y','z')}
        # 마이크
        self.mic_on  = False
        self.mic_buf = deque(maxlen=4096)
        self.mic_stream = None   # sounddevice.Stream

    # ── 작은 토스트+라벨 로그 ─────────────────────────────────
    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ── 권한 체크 ───────────────────────────────────────────
    def _ask_perm(self, *_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            return

        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if hasattr(Permission, "MANAGE_EXTERNAL_STORAGE"):
            need.append(Permission.MANAGE_EXTERNAL_STORAGE)
        if ANDROID_API >= 33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]
        if sd and hasattr(Permission, "RECORD_AUDIO"):
            need.append(Permission.RECORD_AUDIO)

        def _cb(perms, grants):
            ok = any(grants)
            self.btn_sel.disabled = not ok
            if not ok:
                self.log("저장소 권한 거부됨")

        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need, _cb)

    # ── UI 빌드 ──────────────────────────────────────────────
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.label   = Label(text="Pick CSV or use Realtime buttons", size_hint=(1,.09))
        self.btn_sel = Button(text="Select CSV", disabled=True, size_hint=(1,.09),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",   disabled=True, size_hint=(1,.09),
                              on_press=self.run_fft)
        self.btn_rt  = Button(text="Accel FFT (OFF)", size_hint=(1,.09),
                              on_press=self.toggle_realtime)
        self.btn_mic = Button(text="Mic FFT (OFF)", size_hint=(1,.09),
                              on_press=self.toggle_mic,
                              disabled = (sd is None))      # sounddevice 없으면 OFF

        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(self.btn_rt)
        root.add_widget(self.btn_mic)
        root.add_widget(Button(text="EXIT", size_hint=(1,.09), on_press=self.stop))

        self.graph = GraphWidget(size_hint=(1,.55))
        root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ── CSV 파일 선택 ────────────────────────────────────────
    def open_chooser(self, *_):
        # Android 11+ 모든-파일 권한 안내
        if ANDROID and ANDROID_API >= 30:
            from jnius import autoclass
            Env = autoclass("android.os.Environment")
            if not Env.isExternalStorageManager():
                self.log("'모든-파일' 권한을 설정 하세요")
        # SAF 우선
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True, mime_type="text/*")
                return
            except Exception:
                pass
        # 경로 기반 chooser
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True,
                              filters=[("CSV","*.csv")])

    def on_choose(self, sel):
        if not sel:
            return
        self.paths = []
        for raw in sel[:2]:
            p = uri_to_file(raw)
            if not p:
                self.log("복사 실패"); return
            self.paths.append(p)
        self.label.text = " · ".join(os.path.basename(p) for p in self.paths)
        self.btn_run.disabled = False

    # ── CSV → FFT 실행 ──────────────────────────────────────
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_csv_thread, daemon=True).start()

    def _fft_csv_thread(self):
        results = []
        for path in self.paths:
            pts, _, ymax = self.csv_fft(path)
            if pts is None:
                self.log("CSV parse error"); return
            results.append(pts)
        xmax = 50; ymax = max(max(y for _,y in r) for r in results)
        diff = []
        if len(results) == 2:
            a, b = results
            diff = [(a[i][0], abs(a[i][1]-b[i][1]))
                    for i in range(min(len(a), len(b)))]
            ymax = max(ymax, max(y for _,y in diff))
        Clock.schedule_once(lambda *_:
            self.graph.update_graph(results, diff, xmax, ymax))
        Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))

    @staticmethod
    def csv_fft(path):
        try:
            t, a = [], []
            with open(path) as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a) < 2:
                raise ValueError("no samples")
            dt = (t[-1]-t[0]) / len(a)
            f   = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            amp = np.abs(fft(a))[:len(a)//2]
            m   = f <= 50
            s   = np.convolve(amp[m], np.ones(10)/10, 'same')
            return list(zip(f[m], s)), 50, s.max()
        except Exception as e:
            Logger.error(f"FFT csv err: {e}")
            return None,0,0

    # ── 가속도 토글 / FFT ────────────────────────────────────
    def toggle_realtime(self, *_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel, 0)
                threading.Thread(target=self._accel_fft_loop, daemon=True).start()
            except Exception as e:
                self.log(f"Accel err: {e}")
                self.rt_on = False
                self.btn_rt.text = "Accel FFT (OFF)"
        else:
            try: accelerometer.disable()
            except Exception: pass

    def _poll_accel(self, dt):
        if not self.rt_on:
            return False
        try:
            ax, ay, az = accelerometer.acceleration
            if None in (ax,ay,az):
                return
            now = time.time()
            for ax_name, val in zip(('x','y','z'), (ax,ay,az)):
                self.rt_buf[ax_name].append((now, abs(val)))
        except Exception:
            pass

    def _accel_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[ax]) < 64 for ax in ('x','y','z')):
                continue
            data_sets = []
            ymax = 0.0
            for axis in ('x','y','z'):
                ts, vals = zip(*self.rt_buf[axis])
                sig = np.array(vals, float)
                sig -= sig.mean(); sig *= np.hanning(len(sig))
                dt  = (ts[-1]-ts[0])/max(1,len(sig)-1)
                f   = np.fft.fftfreq(len(sig), d=dt)[:len(sig)//2]
                amp = np.abs(fft(sig))[:len(sig)//2]
                m   = f <= 50
                smooth = np.convolve(amp[m], np.ones(8)/8,'same')
                data_sets.append(list(zip(f[m], smooth)))
                ymax = max(ymax, smooth.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(data_sets, [], 50, ymax))

    # ── 마이크 토글/FFT ──────────────────────────────────────
    def toggle_mic(self, *_):
        if sd is None:
            self.log("sounddevice 모듈이 없습니다")
            return
        self.mic_on = not self.mic_on
        self.btn_mic.text = f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self._start_mic()
            except Exception as e:
                self.log(f"mic err: {e}")
                self.mic_on = False
                self.btn_mic.text = "Mic FFT (OFF)"
        else:
            self._stop_mic()

    def _start_mic(self):
        self.mic_buf.clear()
        self.mic_stream = sd.InputStream(samplerate=44100, channels=1,
                                         blocksize=512, dtype='float32',
                                         callback=self._on_mic_block)
        self.mic_stream.start()
        threading.Thread(target=self._mic_fft_loop, daemon=True).start()

    def _stop_mic(self):
        try: self.mic_stream.stop(); self.mic_stream.close()
        except Exception: pass
        self.mic_stream = None

    def _on_mic_block(self, data, frames, time_info, status):
        if not self.mic_on:
            return
        self.mic_buf.extend(data[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.25)
            if len(self.mic_buf) < 2048:
                continue
            sig = np.array(self.mic_buf, float); self.mic_buf.clear()
            sig -= sig.mean(); sig *= np.hanning(len(sig))
            f   = np.fft.fftfreq(len(sig), d=1/44100)[:len(sig)//2]
            amp = np.abs(fft(sig))[:len(sig)//2]
            m   = f <= 1500
            smooth = np.convolve(amp[m], np.ones(16)/16, 'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f[m], smooth))], [],
                                        1500, smooth.max()))

# ── 메인 진입 ──────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
