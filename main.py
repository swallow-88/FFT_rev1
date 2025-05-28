"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판
"""

# ── 표준 및 3rd-party ───────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import numpy as np

from plyer import accelerometer      # 센서
from collections import deque
import queue, time

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
from plyer               import filechooser           # (SAF 실패 시 fallback)
import traceback

DB_REF = 1.0
DB_FLOOR = -120.0
FIXED_DT = 1.0 / 100.0


# ── Android 전용 모듈(있을 때만) ────────────────────────────────────
ANDROID = platform == "android"

toast = None
SharedStorage = None
Permission = None
check_permission = request_permissions = None
ANDROID_API = 0

if ANDROID:
    try:
        from plyer import toast
    except Exception:
        toast = None

    try:
        from androidstorage4kivy import SharedStorage
    except Exception:
        SharedStorage = None

    try:
        from android.permissions import (
            check_permission, request_permissions, Permission)
    except Exception:
        # permissions recipe 가 없는 빌드용 더미
        check_permission = lambda *a, **kw: True
        request_permissions = lambda *a, **kw: None
        class _P:
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
        Permission = _P

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ── 전역 예외 → /sdcard/fft_crash.log ───────────────────────────────
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" +
                     datetime.datetime.now().isoformat() + "\n" + txt + "\n")
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

# ── SAF URI → 앱 캐시 파일 경로 ─────────────────────────────────────
def uri_to_file(u: str) -> str | None:
    if not u:
        return None
    if u.startswith("file://"):
        real = urllib.parse.unquote(u[7:])
        return real if os.path.exists(real) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                u, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None



class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0),(0,1,0),(0,0,1)]
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.5

    MAX_FREQ = 30  # X축 0–50Hz 고정
    RT_BANDS = [0, 5, 10, 20, 50, 100, 150]   # 실시간 VAL 구간

    
    def __init__(self, **kw):
        super().__init__(**kw)
        #self.sample_t0 = time.time()
        #self.sample_count = 0
        self.datasets = []
        self.diff     = []
        self.max_x = self.max_y = 1
        self.min_y = DB_FLOOR
        self.bind(size=lambda *a: Clock.schedule_once(lambda *_: self.redraw(), 0))


    def update_graph(self, ds, df, xm, ym, *, rt: bool):
        """
        rt=True  → 실시간 FFT 모드  
        rt=False → CSV 파일 FFT 모드
        """
        self.rt_mode = rt            # <- 모드 기억
        # ---- 공통 ----
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
    
        # ---- 축 스케일 결정 ----
        self.max_x = 30.0                      # 둘 다 X는 0~30 Hz
        if rt:                      # ── 실시간(VAL) ──
            self.min_y = 0.0
            self.max_y = self.RT_BANDS[-1]   # ==150
        else:                       # ── CSV ──
            self.min_y = 0.0
            self.max_y = max(1e-6, float(ym))
    
        Clock.schedule_once(lambda *_: self.redraw(), 0)

    # -----------------------------------------------
    def _scale(self, pts):
        w = self.width  - 2*self.PAD_X
        h = self.height - 2*self.PAD_Y
        out = []
    
        for x, y in pts:
            # ---------- X ----------
            out.append(self.PAD_X + (x/self.max_x)*w)
    
            # ---------- Y ----------
            if self.rt_mode:                       # 실시간(밴드 매핑)
                edges = self.RT_BANDS
                band_h = h / (len(edges)-1)
    
                # 클리핑
                if y <= edges[0]:
                    y_px = self.PAD_Y
                elif y >= edges[-1]:
                    y_px = self.PAD_Y + h
                else:
                    # y 가 포함된 밴드 찾기
                    for i in range(len(edges)-1):
                        if edges[i] <= y < edges[i+1]:
                            frac = (y - edges[i]) / (edges[i+1]-edges[i])
                            y_px = self.PAD_Y + band_h*i + frac*band_h
                            break
                out.append(y_px)
            else:                                   # CSV – 선형
                out.append(self.PAD_Y + (y/self.max_y)*h)
    
        return out

    def _grid(self):
        Color(.6, .6, .6)
    
        # ── 수평선은 “edge” 개수 - 1 ──
        edges = self.RT_BANDS if self.rt_mode else \
                [self.max_y * i/5 for i in range(6)]
    
        for val in edges:
            y = self._val_to_y(val)
            Line(points=[self.PAD_X, y, self.width-self.PAD_X, y])
    
        # ── 수직선은 고정 0·5·…·30 Hz ──
        gx = (self.width-2*self.PAD_X) / 6
        for i in range(7):
            x = self.PAD_X + gx*i
            Line(points=[x, self.PAD_Y, x, self.height-self.PAD_Y])

    def _labels(self):
        # 기존 라벨 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)
    
        # ── X 축 (0·5·…·30 Hz) ───────────────────────────────
        for i in range(7):
            f = 5*i
            xpos = self.PAD_X + (self.width-2*self.PAD_X)*(f/self.MAX_FREQ) - 18
            lbl = Label(text=f"{f} Hz", size_hint=(None,None),
                        size=(50,20), pos=(xpos, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)
    
        # ── Y 축 (왼쪽) ───────────────────────────────────────
        # -------- Y 라벨 --------
        if self.rt_mode:
            edges = self.RT_BANDS
        else:
            edges = [self.max_y*i/5 for i in range(6)]
    
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            y_mid  = self._val_to_y((lo+hi)/2) - 8
            lbl = Label(text=f"{lo:.0f}–{hi:.0f}",
                        size_hint=(None,None), size=(90,20),
                        pos=(self.PAD_X-50, y_mid))   # ←5px 오른쪽
            lbl._axis = True
            self.add_widget(lbl)
    # ── 1) GraphWidget.redraw – 들여쓰기 정리 ──────────────────────────
    def redraw(self, *_):
        self.clear_widgets()
        try:
            if self.width <= 2*self.PAD_X or self.height <= 2*self.PAD_Y:
                return
    
            self.clear_widgets()
            self.canvas.clear()
            if not self.datasets:
                return
    
            peaks = []                                        # ← 들여쓰기 OK
            with self.canvas:
                self._grid()
                self._labels()
    
                for idx, pts in enumerate(self.datasets):
                    if not pts:
                        continue
    
                    # 실시간 모드일 때 y 클립
                    if getattr(self, "rt_mode", False):
                        pts = [(x, min(y, self.max_y)) for x, y in pts]
    
                    Color(*self.COLORS[idx % len(self.COLORS)])
                    Line(points=self._scale(pts), width=self.LINE_W)
    
                    fx, fy = max(pts, key=lambda p: p[1])
                    sx, sy = self._scale([(fx, fy)])[0:2]
                    peaks.append((fx, fy, sx, sy))
    
                if self.diff:
                    Color(*self.DIFF_CLR)
                    Line(points=self._scale(self.diff), width=self.LINE_W)
    
            # ---------- 피크 라벨 ----------
            for fx, fy, sx, sy in peaks:                      # peaks 로!
                lbl = Label(text=f"▲ {fx:.1f} Hz  {fy:.0f}",
                            size_hint=(None, None), size=(110, 22),
                            pos=(int(sx-40), int(sy+6)))
                lbl._peak = True
                self.add_widget(lbl)
    
            # ---------- Δ 표시 ----------
            if len(peaks) >= 2:
                delta = abs(peaks[0][0] - peaks[1][0])
                bad   = delta > 1.5
                clr   = (1,0,0,1) if bad else (0,1,0,1)
                info = Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                             size_hint=(None, None), size=(190, 24),
                             pos=(int(self.PAD_X),
                                  int(self.height - self.PAD_Y + 6)),
                             color=clr)
                info._peak = True
                self.add_widget(info)
    
        except Exception as e:                                 # ← try 와 같은 칸
            _dump_crash(f"redraw error: {e}\n{traceback.format_exc()}")


    def _val_to_y(self, v):
        h = self.height - 2*self.PAD_Y
        if self.rt_mode:               # 고정 밴드(log형)
            edges = self.RT_BANDS
            band_h = h / (len(edges)-1)
            if v <= edges[0]:
                return self.PAD_Y
            if v >= edges[-1]:
                return self.PAD_Y + h
            for i in range(len(edges)-1):
                if edges[i] <= v < edges[i+1]:
                    frac = (v-edges[i])/(edges[i+1]-edges[i])
                    return self.PAD_Y + band_h*i + frac*band_h
        else:                           # 선형
            return self.PAD_Y + h * (v/self.max_y)
        
# ── 메인 앱 ───────────────────────────────────────────────────────
class FFTApp(App):
    RT_WIN   = 256
    FIXED_DT = 1.0 / 60.0
    MIN_FREQ = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ① 실시간 토글 플래그를 미리 초기화
        self.rt_on = False
        # ② 가속도 버퍼 준비
        self.rt_buf = {
            'x': deque(maxlen=self.RT_WIN),
            'y': deque(maxlen=self.RT_WIN),
            'z': deque(maxlen=self.RT_WIN),
        }
# 3) FFTApp.__init__
        self.sample_t0 = time.time()
        self.sample_count = 0

    
    # ── 작은 토스트+라벨 로그 ─────────────────────────────────────
    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception:
                pass

    # ── 저장소 권한 요청 ────────────────────────────────────────
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage:           # SAF만 쓰면 file 권한 불필요
            self.btn_sel.disabled = False
            return

        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        MANAGE = getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)
        if MANAGE:
            need.append(MANAGE)
        if ANDROID_API >= 33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]

        def _cb(perms, grants):
            self.btn_sel.disabled = not any(grants)
            if not any(grants):
                self.log("저장소 권한 거부됨 – CSV 파일을 열 수 없습니다")

        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need, _cb)


    
    # ---------- ① 토글  ----------
    # ---------- 실시간 FFT 토글 ----------
    def toggle_realtime(self, *_):
        """
        ▶ 버튼을 누를 때마다 실시간 가속도 FFT ON/OFF 전환.
        · Android 기기라면 가능한 한 빠른 센서 주기로 등록한다.
        · 토글 ON 시 → 센서 enable + polling·FFT 스레드 시작
        · 토글 OFF 시 → 센서 disable + 스레드 자동 종료
        """
        # 상태 반전
        self.rt_on = not getattr(self, "rt_on", False)
        self.btn_rt.text = f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"

        if self.rt_on:
            # ── 1) 센서 최대 속도로 등록 (Android 전용) ──────────────────
            if ANDROID:
                try:
                    from jnius import autoclass, cast
                    PythonActivity = autoclass("org.kivy.android.PythonActivity")
                    Context        = cast("android.content.Context",
                                          PythonActivity.mActivity)
                    SensorManager  = autoclass("android.hardware.SensorManager")
                    sm     = Context.getSystemService(Context.SENSOR_SERVICE)
                    accel  = sm.getDefaultSensor(SensorManager.SENSOR_ACCELEROMETER)
                    # SENSOR_DELAY_FASTEST == 0
                    sm.registerListener(PythonActivity.mActivity, accel,
                                        SensorManager.SENSOR_DELAY_FASTEST)
                except Exception as e:
                    Logger.warning(f"FASTEST sensor register fail: {e}")

            # ── 2) plyer accelerometer on ───────────────────────────────
            try:
                accelerometer.enable()
            except (NotImplementedError, Exception) as e:
                self.log(f"센서 사용 불가: {e}")
                self.rt_on = False
                self.btn_rt.text = "Realtime FFT (OFF)"
                return

            # ── 3) Kivy Clock 로 polling, 별도 스레드로 FFT ───────────
            Clock.schedule_interval(self._poll_accel, 0)
            threading.Thread(target=self._rt_fft_loop, daemon=True).start()

        else:
            # ── OFF : 센서·Clock·스레드 정리 ────────────────────────────
            try:
                accelerometer.disable()
            except Exception:
                pass
            # Clock.schedule_interval 에서 _poll_accel 이 False 반환 → 자동 해제
        
    # ---------- ② 센서 polling ----------
    def _poll_accel(self, dt):
        """
        매 프레임마다 센서를 읽어 각 축별 deque 에
        (timestamp, 절대값(가속도)) 튜플을 저장.
        """
        if not self.rt_on:
            return False  # Clock 에서 해제

        try:
            ax, ay, az = accelerometer.acceleration
            if None in (ax, ay, az):
                return
            now = time.time()
            self.rt_buf['x'].append((now, abs(ax)))
            self.rt_buf['y'].append((now, abs(ay)))
            self.rt_buf['z'].append((now, abs(az)))
        except Exception as e:
            Logger.warning(f"accel read fail: {e}")
            
        self.sample_count += 1
        if time.time() - self.sample_t0 >= 1.0:          # 1초마다
            fs = self.sample_count / (time.time() - self.sample_t0)
            if fs < 100:
                self.log(f"⚠️ 샘플 속도 {fs:.0f} Hz → 50 Hz 분석 불완전")
            self.sample_t0 = time.time()
            self.sample_count = 0
    
    # ---------- ③ FFT 백그라운드 ----------# ── 2) _rt_fft_loop – dt를 실측으로 계산 ───────────────────────────
    def _rt_fft_loop(self):
        while self.rt_on:
            try:                                  # ① try: 블록 시작
                time.sleep(0.5)
    
                # 버퍼가 아직 다 안 찼으면 건너뜀
                if any(len(self.rt_buf[ax]) < self.RT_WIN for ax in ('x', 'y', 'z')):
                    continue
    
                datasets = []                     # 세미콜론(;) 제거
                ymax = 0.0
    
                for axis in ('x', 'y', 'z'):
                    ts, vals = zip(*self.rt_buf[axis])
                    n   = self.RT_WIN
                    sig = np.asarray(vals, float) * np.hanning(n)
    
                    dt   = FFTApp.FIXED_DT                   # 100 Hz 고정
                    freq = np.fft.fftfreq(n, d=dt)[:n//2]
                    amp  = np.abs(fft(sig))[:n//2]          # VAL 스펙트럼
    
                    mask   = (freq <= self.graph.MAX_FREQ) & (freq >= self.MIN_FREQ)
                    freq   = freq[mask]
                    smooth = np.convolve(amp[mask], np.ones(8)/8, 'same')
                    smooth = np.clip(smooth, 0.0, 150.0)   # 0–150 VAL
                    # _rt_fft_loop → smooth 계산 직후
                    
                    
                    datasets.append(list(zip(freq, smooth)))
                    ymax = max(ymax, smooth.max())
                


                # x축은 0-30 Hz 고정 → xm 인자는 30 고정
                # _rt_fft_loop() 마지막 쪽
                Clock.schedule_once(
                    lambda *_: self.graph.update_graph(datasets, [], 30, ymax, rt=True)
                )
    
            except Exception as e:                # ② try 와 같은 들여쓰기
                _dump_crash(f"_rt_fft_loop error: {e}\n{traceback.format_exc()}")
                continue
    # ── UI 구성 ────────────────────────────────────────────────
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

        # build() 안 – EXIT 버튼 위쪽에 추가, 실시간 가속도 분석을 위해
        self.btn_rt  = Button(text="Realtime FFT (OFF)", size_hint=(1,.1),
                              on_press=self.toggle_realtime)
        root.add_widget(self.btn_rt)

        
        self.graph = GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ── 파일 선택 ──────────────────────────────────────────────
    def open_chooser(self,*_):

        # Android 11+ : ‘모든 파일’ 권한 안내
        if ANDROID and ANDROID_API >= 30:
            try:
                from jnius import autoclass
                Env = autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():
                    mv = ModalView(size_hint=(.8,.35))
                    box=BoxLayout(orientation='vertical', spacing=10, padding=10)
                    box.add_widget(Label(
                        text="⚠️ CSV 파일에 접근하려면\n'모든 파일' 권한이 필요합니다.",
                        halign="center"))
                    box.add_widget(Button(text="권한 설정으로 이동",
                                          size_hint=(1,.4),
                                          on_press=lambda *_: (
                                              mv.dismiss(),
                                              self._goto_allfiles_permission())))
                    mv.add_widget(box); mv.open()
                    return
            except Exception:
                Logger.exception("ALL-FILES check 오류(무시)")

        # ① SAF picker (권장) ------------------------------------
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(
                    callback=self.on_choose,
                    multiple=True,
                    mime_type="text/*")
                return
            except Exception as e:
                Logger.exception("SAF picker fail")
                self.log(f"SAF 선택기 오류: {e}")

        # ② 경로 기반 chooser -----------------------------------
        try:
            filechooser.open_file(
                on_selection=self.on_choose,      # ★ 키워드 인자!
                multiple=True,
                filters=[("CSV","*.csv")],
                native=False,
                path="/storage/emulated/0/Download")
        except Exception as e:
            Logger.exception("legacy chooser fail")
            self.log(f"파일 선택기를 열 수 없습니다: {e}")

    def _goto_allfiles_permission(self):
        from jnius import autoclass
        Intent   = autoclass("android.content.Intent")
        Settings = autoclass("android.provider.Settings")
        Uri      = autoclass("android.net.Uri")
        act      = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(
            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))

    # ── 파일 선택 결과 콜백 ─────────────────────────────────────
    def on_choose(self, sel):
        Logger.info(f"선택: {sel}")
        if not sel:
            return
        paths=[]
        for raw in sel[:2]:
            real = uri_to_file(raw)
            Logger.info(f"{raw} → {real}")
            if not real:
                self.log("❌ 복사 실패"); return
            paths.append(real)

        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # ── FFT 실행 ──────────────────────────────────────────────
    def run_fft(self,*_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res = []
        for p in self.paths:
            pts, xm, ym = self.csv_fft(p)
            if pts is None:
                self.log("CSV parse err"); return
            res.append((pts, xm, ym))
    
        if len(res) == 1:
            pts, xm, ym = res[0]
            Clock.schedule_once(
                lambda *_: self.graph.update_graph([pts], [], xm, ym, rt=False)
            )

 
        else:
            (f1, x1, y1), (f2, x2, y2) = res
            diff = [(f1[i][0], abs(f1[i][1]-f2[i][1]))
                    for i in range(min(len(f1), len(f2)))]
            xm = max(x1, x2)
            ym = max(y1, y2, max(y for _, y in diff))
            Clock.schedule_once(
                lambda *_: self.graph.update_graph([f1, f2], diff, xm, ym, rt=False)              
            )
            
        Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))

    # ── CSV → FFT ────────────────────────────────────────────
    @staticmethod
    def csv_fft(path: str):
        """
        CSV( time[s] , value ) → 0-30 Hz 진폭(VAL) 스펙트럼을 돌려준다.
        반환값: ([(freq, val), …],   30,   val_max)
        """
        try:
            t, a = [], []
            with open(path, newline="") as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0]));  a.append(float(r[1]))
                    except Exception:
                        pass
            if len(a) < 2:
                raise ValueError("too few samples")
    
              # 고정 100 Hz 샘플 주기
             # ① 샘플 간 평균 Δt 계산 ― 실제 CSV 주기 사용
            dt   = (t[-1] - t[0]) / (len(t) - 1)  # sec/샘플
            if dt <= 0:             # 파일이 잘못된 경우 방어
                raise ValueError("invalid time column")
            sig  = np.asarray(a) * np.hanning(len(a))
            amp  = np.abs(fft(sig))[:len(a)//2]   # ← VAL 진폭
            freq = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
    
            # ② Nyquist 주파수보다 낮도록, 30 Hz 보다 작은 경우엔 최대치까지만
            f_hi   = min(30, freq.max())
            mask   = (freq <= f_hi) & (freq >= 1)
            freq   = freq[mask]
            smooth = np.convolve(amp[mask], np.ones(10)/10, 'same')
    
            return list(zip(freq, smooth)), 30, float(smooth.max())
    
        except Exception as e:
            Logger.error(f"csv_fft err {e}")
            return None, 0, 0
# ── 실행 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
