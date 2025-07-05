"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 - 통합 안정판
+ 10/30/60/120 s 실시간 가속도 기록 (Downloads 폴더 저장)
+ X/Y/Z 세로 3 분할 그래프  - 각 축 RMS(실선)·Peak(점선) 표시
+ CSV 최대 3 개(x / y / z) 선택, ΔF 비교·실시간 공진수 추적
─────────────────────────────────────────────────────────────
변경 핵심
 1) self.graph ➜ self.graphs[0|1|2]  - 축별 위젯 분리
 2) _rt_fft_loop / _fft_bg  데이터→그래프 매핑 전면 수정
 3) 버퍼 공유용 self._buf_lock 추가 (스레드 안정)
 4) DOWNLOAD_DIR 빈 문자열 Fallback 보강
 5) 미사용 _record_poll 제거 + 코드 전반 소규모 정리
"""

import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time, re
import numpy as np
from collections import deque
from numpy.fft import fft
from plyer import accelerometer
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.graphics import Line, Color
from kivy.utils import platform
from plyer import filechooser
from kivy.uix.spinner import Spinner
# ------------------------------------------------------------------
#                     ★ ① 사용자 조정 상수 ★
# ------------------------------------------------------------------
BAND_HZ = 0.5           # FFT 밴드 폭
REF_MM_S, REF_ACC = 0.01, 0.981
MEAS_MODE = "VEL"       # "VEL" 또는 "ACC"
SMOOTH_N = 1            # RMS 스무딩 창
HPF_CUTOFF, MAX_FMAX = 5.0, 200
REC_DURATION_DEFAULT = 60.0
FN_BAND = (5, 50)       # 공진 탐색 범위
BUF_LEN, MIN_LEN = 16384, 256
# ------------------------------------------------------------------
#                    ★ ② Android 전용 준비 ★
# ------------------------------------------------------------------
ANDROID = platform == "android"
toast = SharedStorage = Permission = None
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
        Permission = type("P", (), {})
        check_permission = lambda *a, **kw: True
        request_permissions = lambda *a, **kw: None
    try:                                   # Downloads 절대경로 확보
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
        Environment = autoclass("android.os.Environment")
        DOWNLOAD_DIR = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() or "/sdcard/Download"
    except Exception:
        DOWNLOAD_DIR = "/sdcard/Download"
else:
    DOWNLOAD_DIR = os.path.expanduser("~/Download")
# ------------------------------------------------------------------
#                    ★ ③ 공용 함수/도우미 ★
# ------------------------------------------------------------------
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "=" * 60 + "\n" +
                     datetime.datetime.now().isoformat() + "\n" + txt + "\n")
    except Exception:
        pass
    Logger.error(txt)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)), size_hint=(.9, .9)).open())
sys.excepthook = _ex
# ..................................................................
def uri_to_file(u: str) -> str | None:
    if not u:
        return None
    if u.startswith("file://"):
        real = urllib.parse.unquote(u[7:])
        return real if os.path.exists(real) else None
    if u.startswith("content://") and ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                u, uuid.uuid4().hex + ".csv", to_downloads=True)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
            return None
    if os.path.exists(u):
        return u
    Logger.warning(f"uri_to_file: cannot access {u}")
    return "NO_PERMISSION"
# ..................................................................
def acc_to_spec(freq, amp_a):
    if MEAS_MODE == "VEL":
        f_nz = np.where(freq < 1e-6, 1e-6, freq)
        amp = amp_a / (2 * np.pi * f_nz) * 1e3
        ref = REF_MM_S
    else:
        amp, ref = amp_a, REF_ACC
    return amp, ref
# ..................................................................
def smooth_y(vals, n=None):
    n = n or SMOOTH_N
    if n <= 1 or len(vals) < n:
        return vals[:]
    return np.convolve(vals, np.ones(n) / n, mode="same")
# ..................................................................
def welch_band_stats(sig, fs, f_lo=HPF_CUTOFF, f_hi=MAX_FMAX,
                     band_w=BAND_HZ, seg_n=None, overlap=0.5):
    seg_n = seg_n or int(fs * 4)
    step, win = int(seg_n * (1 - overlap)), np.hanning(seg_n)
    spec_sum, ptr = None, 0
    while ptr + seg_n <= len(sig):
        seg = (sig[ptr:ptr + seg_n] - sig[ptr:ptr + seg_n].mean()) * win
        ps = (abs(np.fft.rfft(seg)) ** 2) / (np.sum(win ** 2) * fs)
        spec_sum = ps if spec_sum is None else spec_sum + ps
        ptr += step
    if spec_sum is None:
        return [], []
    psd = spec_sum / ((ptr - step) // step + 1)
    freq = np.fft.rfftfreq(seg_n, d=1 / fs)
    msel = (freq >= f_lo) & (freq <= f_hi)
    freq, psd = freq[msel], psd[msel]
    amp_lin, REF0 = acc_to_spec(freq, np.sqrt(psd * 2))
    band_rms, band_pk = [], []
    for lo in np.arange(f_lo, f_hi, band_w):
        hi = lo + band_w
        s = (freq >= lo) & (freq < hi)
        if not s.any():
            continue
        rms = np.sqrt(np.mean(amp_lin[s] ** 2))
        pk = amp_lin[s].max()
        cen = (lo + hi) / 2
        band_rms.append((cen, 20 * np.log10(max(rms, REF0 * 1e-4) / REF0)))
        band_pk.append((cen, 20 * np.log10(max(pk, REF0 * 1e-4) / REF0)))
    if len(band_rms) >= SMOOTH_N:
        ys = smooth_y([y for _, y in band_rms])
        band_rms = list(zip([x for x, _ in band_rms], ys))
    return band_rms, band_pk
# ------------------------------------------------------------------
#                    ★ ④ 그래프 위젯 정의 ★
# ------------------------------------------------------------------
class GraphWidget(Widget):
    PAD_X, PAD_Y, LINE_W = 80, 50, 2.5
    COLORS = [(1, 0, 0), (1, 1, 0), (0, 0, 1),
              (0, 1, 1), (0, 1, 0), (1, 0, 1)]
    DIFF_CLR = (1, 1, 1)
    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff, self.max_x = [], [], 1
        self.Y_MIN, self.Y_MAX, self.Y_TICKS = 0, 100, [0, 20, 40, 60, 80, 100]
        self.bind(size=self.redraw)

    # ..............................................
    def update_graph(self, ds, df, xm):
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff = df or []
        self.max_x = max(1e-6, float(xm))
        ys = [y for seq in self.datasets + [self.diff] for _, y in seq]
        if ys:
            top = ((int(max(ys)) // 20) + 1) * 20
            low = ((int(min(ys)) // 20) - 1) * 20
            self.Y_MIN, self.Y_MAX = low, top
            self.Y_TICKS = list(range(low, top + 1, 20))
        self.redraw()
    # ..............................................
    def y_pos(self, v):
        h = self.height - 2 * self.PAD_Y
        return self.PAD_Y + (v - self.Y_MIN) / (self.Y_MAX - self.Y_MIN) * h
    # ..............................................
    def _scale(self, pts):
        w = self.width - 2 * self.PAD_X
        return [self.PAD_X + x / self.max_x * w if i % 2 == 0 else
                self.y_pos(y) for i, (x, y) in enumerate(pts * 1)]
    # ..............................................
    def _grid(self):
        n_tick = int(self.max_x // 10) + 1
        gx = (self.width - 2 * self.PAD_X) / max(n_tick - 1, 1)
        Color(.6, .6, .6)
        for i in range(n_tick):
            Line(points=[self.PAD_X + i * gx, self.PAD_Y,
                         self.PAD_X + i * gx, self.height - self.PAD_Y])
        for v in self.Y_TICKS:
            Line(points=[self.PAD_X, self.y_pos(v),
                         self.width - self.PAD_X, self.y_pos(v)])

    def _clear_labels(self):
        for w in list(self.children):
            if getattr(w, "_axis", False) or getattr(w, "_peak", False):
                self.remove_widget(w)
    # ..............................................
    def redraw(self, *_):
        self.canvas.clear()
        self._clear_labels()
        if not self.datasets:
            return
        with self.canvas:
            self._grid()
            peaks = []
            for idx, pts in enumerate(self.datasets):
                if not pts:
                    continue
                Color(*self.COLORS[idx // 2 % len(self.COLORS)])
                scaled = self._scale(pts)
                if idx % 2:   # Peak
                    dash, gap = 10, 6
                    for i in range(0, len(scaled) - 2, 2):
                        x1, y1, x2, y2 = scaled[i:i + 4]
                        seg = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        nx, ny, s, draw = (x2 - x1) / seg, (y2 - y1) / seg, 0.0, True
                        while s < seg:
                            l = min(dash if draw else gap, seg - s)
                            if draw:
                                Line(points=[x1 + nx * s, y1 + ny * s,
                                             x1 + nx * (s + l), y1 + ny * (s + l)],
                                     width=self.LINE_W)
                            s, draw = s + l, not draw
                else:         # RMS
                    Line(points=scaled, width=self.LINE_W)
                    fx, fy = max(pts, key=lambda p: p[1])
                    sx, sy = self._scale([(fx, fy)])[0:2]
                    peaks.append((fx, fy, sx, sy))
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz", size_hint=(None, None),
                        size=(90, 24), pos=(sx - 30, sy + 6))
            lbl._peak = True
            self.add_widget(lbl)
# ------------------------------------------------------------------
#                    ★ ⑤ 메인 앱 클래스 ★
# ------------------------------------------------------------------
class FFTApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.rt_on, self.rec_on = False, False
        self.rt_buf = {ax: deque(maxlen=BUF_LEN) for ax in "xyz"}
        self._buf_lock = threading.Lock()
        self.rec_start, self.rec_files = 0.0, {}
        self.REC_DURATION = REC_DURATION_DEFAULT
        self.last_fn, self.F0 = None, None

    # ..............................................................
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try:
                toast.toast(msg)
            except Exception:
                pass
    # ..............................................................
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        # ── 안내 라벨
        self.label = Label(text="Pick up to 3 CSV (x/y/z)", size_hint=(1, .05))
        root.add_widget(self.label)
        # ── 버튼 3 개
        self.btn_sel = Button(text="Select CSV", size_hint=(1, .05),
                              disabled=True, on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN", size_hint=(1, .05),
                              disabled=True, on_press=self.run_fft)
        self.btn_rec = Button(text=f"Record {int(self.REC_DURATION)} s",
                              size_hint=(1, .05), disabled=True,
                              on_press=self.start_recording)
        root.add_widget(self.btn_sel), root.add_widget(self.btn_run), root.add_widget(self.btn_rec)
        # ── Spinner 3 종
        self.spin_dur = Spinner(text=f"{int(self.REC_DURATION)} s",
                                values=("10 s", "30 s", "60 s", "120 s"),
                                size_hint=(1, .05))
        self.spin_dur.bind(text=lambda s, t: self._set_rec_dur(float(t.split()[0])))
        self.spin_sm = Spinner(text=str(SMOOTH_N), values=("1", "2", "3", "4", "5"),
                               size_hint=(1, .05))
        self.spin_sm.bind(text=lambda s, t: self._set_smooth(int(t)))
        root.add_widget(self.spin_dur), root.add_widget(self.spin_sm)
        # ── 모드·F₀·Realtime
        self.btn_mode = Button(text=f"Mode: {MEAS_MODE}", size_hint=(1, .05),
                               on_press=self._toggle_mode)
        self.btn_setF0 = Button(text="Set F₀ (baseline)", size_hint=(1, .05),
                                on_press=self._save_baseline)
        self.btn_rt = Button(text="Realtime FFT (OFF)", size_hint=(1, .05),
                             on_press=self.toggle_realtime)
        root.add_widget(self.btn_mode), root.add_widget(self.btn_setF0), root.add_widget(self.btn_rt)
        # ── 그래프 3 칸
        self.graphs = []
        gbox = BoxLayout(orientation="vertical", size_hint=(1, .60), spacing=4)
        for _ in range(3):
            gw = GraphWidget(size_hint=(1, 1 / 3))
            self.graphs.append(gw)
            gbox.add_widget(gw)
        root.add_widget(gbox)
        Clock.schedule_once(self._ask_perm, 0)
        return root
    # ..............................................................
    def _set_rec_dur(self, sec):
        self.REC_DURATION = sec
        self.btn_rec.text = f"Record {int(sec)} s"
    def _set_smooth(self, n):
        global SMOOTH_N
        SMOOTH_N = n
        self.log(f"▶ 스무딩 창 = {SMOOTH_N}")
    # ..............................................................
    def _ask_perm(self, *_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = self.btn_rec.disabled = False
            return
        need = [getattr(Permission, "READ_EXTERNAL_STORAGE", ""),
                getattr(Permission, "WRITE_EXTERNAL_STORAGE", "")]
        if (MANAGE := getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)):
            need.append(MANAGE)
        if ANDROID_API >= 33:
            need += [getattr(Permission, n, "") for n in
                     ("READ_MEDIA_IMAGES", "READ_MEDIA_AUDIO", "READ_MEDIA_VIDEO")]
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = self.btn_rec.disabled = False
        else:
            request_permissions(need, lambda p, g:
                setattr(self.btn_sel, "disabled", not any(g)) or
                setattr(self.btn_rec, "disabled", not any(g)))
    # ------------------------------------------------------------------
    #                    ★ ⑤-1  녹음 루틴  ★
    # ------------------------------------------------------------------
    def start_recording(self, *_):
        if self.rec_on:
            self.log("이미 기록 중입니다"); return
        try:
            accelerometer.enable()
        except Exception as e:
            self.log(f"센서 사용 불가: {e}"); return
        if not self.rt_on:
            self.toggle_realtime()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_files = {}
        try:
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            for ax in "xyz":
                fp = open(os.path.join(DOWNLOAD_DIR, f"acc_{ax}_{ts}.csv"),
                          "w", newline="", encoding="utf-8")
                csv.writer(fp).writerow(("time", "acc"))
                self.rec_files[ax] = fp
        except Exception as e:
            self.log(f"파일 열기 실패: {e}")
            return
        self.rec_on, self.rec_start = True, time.time()
        self.btn_rec.disabled = True
        self.label.text = f"Recording 0/{int(self.REC_DURATION)} s …"
        Clock.schedule_once(self._stop_recording, self.REC_DURATION)
    # ------------------------------------------------------------------
    def _stop_recording(self, *_):
        if not self.rec_on:
            return
        for fp in self.rec_files.values():
            try: fp.close()
            except Exception: pass
        self.rec_files.clear()
        self.rec_on, self.btn_rec.disabled = False, False
        self.log("✅ Recording complete!")
    # ------------------------------------------------------------------
    def toggle_realtime(self, *_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
            except Exception as e:
                self.log(f"센서 사용 불가: {e}")
                self.rt_on = False
                self.btn_rt.text = "Realtime FFT (OFF)"
                return
            Clock.schedule_interval(self._poll_accel, 0)
            threading.Thread(target=self._rt_fft_loop, daemon=True).start()
        else:
            try:
                accelerometer.disable()
            except Exception:
                pass
    # ------------------------------------------------------------------
    def _poll_accel(self, dt):
        if not self.rt_on:
            return False
        try:
            ax, ay, az = accelerometer.acceleration
            if None in (ax, ay, az):
                return
            now = time.time()
            with self._buf_lock:
                for axis, val in zip("xyz", (abs(ax), abs(ay), abs(az))):
                    prev = self.rt_buf[axis][-1][0] if self.rt_buf[axis] else now - dt
                    self.rt_buf[axis].append((now, val, now - prev))
            if self.rec_on:
                rel = now - self.rec_start
                for a, v in zip("xyz", (ax, ay, az)):
                    csv.writer(self.rec_files[a]).writerow((rel, v))
                if int(rel * 2) % 2 == 0:
                    self.label.text = f"Recording {rel:4.1f}/{int(self.REC_DURATION)} s …"
        except Exception as e:
            Logger.warning(f"acc read fail: {e}")
    # ------------------------------------------------------------------
    def _rt_fft_loop(self):
        try:
            while self.rt_on:
                time.sleep(0.5)
                with self._buf_lock:
                    if any(len(self.rt_buf[a]) < MIN_LEN for a in "xyz"):
                        continue
                    buf_copy = {a: list(self.rt_buf[a]) for a in "xyz"}
                axis_sets, xmax = {}, 0
                for axis in "xyz":
                    ts, val, dt_arr = zip(*buf_copy[axis])
                    fs = 1.0 / np.median(dt_arr[-512:])
                    f_hi = min(fs * 0.5, MAX_FMAX)
                    if f_hi < HPF_CUTOFF + BAND_HZ:
                        continue
                    band_rms, band_pk = welch_band_stats(
                        np.asarray(val), fs, HPF_CUTOFF, f_hi, BAND_HZ)
                    if not band_rms:
                        continue
                    axis_sets[axis] = (band_rms, band_pk)
                    xmax = max(xmax, f_hi)
                    # 공진수 추적
                    lo, hi = FN_BAND
                    freqs = np.array([x for x, _ in band_rms])
                    mags = np.array([y for _, y in band_rms])
                    s = (freqs >= lo) & (freqs <= hi)
                    if s.any():
                        self.last_fn = freqs[s][mags[s].argmax()]
                # 그래프 갱신
                if axis_sets:
                    def _update(*_):
                        for idx, axis in enumerate("xyz"):
                            data = axis_sets.get(axis, ([], []))
                            self.graphs[idx].update_graph(
                                list(data), [], xmax)
                    Clock.schedule_once(_update)
        except Exception:
            Logger.exception("Realtime thread crashed")
            self.rt_on = False
            Clock.schedule_once(lambda *_:
                setattr(self.btn_rt, "text", "Realtime FFT (OFF)"))
    # ------------------------------------------------------------------
    #                    ★ ⑤-2  CSV-FFT 루틴 ★
    # ------------------------------------------------------------------
    def open_chooser(self, *_):
        if ANDROID and ANDROID_API >= 30 and not self._has_allfiles_perm():
            try:
                from jnius import autoclass
                Env = autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():
                    mv = ModalView(size_hint=(.8, .35))
                    box = BoxLayout(orientation="vertical", spacing=10, padding=10)
                    box.add_widget(Label(
                        text="📂 ‘모든-파일’ 권한이 필요합니다.", halign="center"))
                    box.add_widget(Button(text="설정 열기", size_hint=(1, .4),
                        on_press=lambda *_: (mv.dismiss(), self._goto_allfiles_permission())))
                    mv.add_widget(box); mv.open(); return
            except Exception:
                Logger.exception("perm-check")
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True, mime_type="text/*"); return
            except Exception:
                pass
        filechooser.open_file(on_selection=self.on_choose, multiple=True,
                              filters=[("CSV", "*.csv")], path=DOWNLOAD_DIR)
    # ..............................................................
    def on_choose(self, sel, *_):
        if not sel: return
        self.paths = []
        for raw in sel[:3]:
            real = uri_to_file(raw)
            if real == "NO_PERMISSION":
                self.log("❌ 권한 없음 – SAF Picker 로 시도해 주세요"); return
            if not real:
                self.log("❌ 파일 복사 실패"); return
            self.paths.append(real)
        self.label.text = " · ".join(os.path.basename(p) for p in self.paths)
        self.btn_run.disabled = False
    # ..............................................................
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()
    # ..............................................................
    def _fft_bg(self):
        try:
            axis_map = {"x": 0, "y": 1, "z": 2}
            graph_data = {0: ([], []), 1: ([], []), 2: ([], [])}
            xmax = 0
            for path in self.paths:
                t, a = self._load_csv(path)
                if t is None:
                    raise ValueError(f"{os.path.basename(path)}: CSV parse fail")
                dt = np.median(np.diff(t)); nyq = 0.5 / dt
                FMAX = int(min(nyq, MAX_FMAX))
                if FMAX < HPF_CUTOFF + BAND_HZ:
                    raise ValueError("sample-rate too low")
                sig = (a - a.mean()) * np.hanning(len(a))
                raw = np.fft.fft(sig)
                amp_a = 2 * abs(raw[:len(a) // 2]) / (len(a) * np.sqrt(2))
                freq = np.fft.fftfreq(len(a), d=dt)[:len(a) // 2]
                sel = (freq >= HPF_CUTOFF) & (freq <= FMAX)
                freq, amp_a = freq[sel], amp_a[sel]
                amp_lin, REF0 = acc_to_spec(freq, amp_a)
                rms, pk = [], []
                for lo in np.arange(HPF_CUTOFF, FMAX, BAND_HZ):
                    hi = lo + BAND_HZ
                    m = (freq >= lo) & (freq < hi)
                    if not m.any(): continue
                    cen = (lo + hi) / 2
                    rms.append((cen, 20 * np.log10(
                        max(np.sqrt(np.mean(amp_lin[m] ** 2)), REF0 * 1e-4) / REF0)))
                    pk.append((cen, 20 * np.log10(
                        max(amp_lin[m].max(), REF0 * 1e-4) / REF0)))
                if len(rms) >= SMOOTH_N:
                    rms = list(zip([x for x, _ in rms],
                                   smooth_y([y for _, y in rms])))
                # ---------------- axis 결정 ----------------
                idx = axis_map.get(re.search(r"_([xyz])_", path).group(1), 0) \
                    if re.search(r"_([xyz])_", path) else 0
                graph_data[idx] = (rms, pk)
                xmax = max(xmax, FMAX)
            # ---------------- UI 스케줄 ----------------
            def _update(*_):
                for i, (rms, pk) in graph_data.items():
                    self.graphs[i].update_graph([rms, pk], [], xmax)
            Clock.schedule_once(_update)
        except Exception as e:
            Clock.schedule_once(lambda *_: self.log(f"FFT 오류: {e}"))
        finally:
            Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))
    # ..............................................................
    def _load_csv(self, path):
        num_re = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?$")
        try:
            t, a = [], []
            with open(path, encoding="utf-8", errors="replace") as f:
                sample = f.read(1024); f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
                except csv.Error:
                    dialect = csv.get_dialect("excel")
                for row in csv.reader(f, dialect):
                    if len(row) < 2: continue
                    if not (num_re.match(row[0].strip())
                            and num_re.match(row[1].strip())): continue
                    t.append(float(row[0].replace(",", ".")))
                    a.append(float(row[1].replace(",", ".")))
            return (None, None) if len(a) < 2 else (np.asarray(t, float), np.asarray(a, float))
        except Exception as e:
            Logger.error(f"CSV read err: {e}")
            return None, None
    # ------------------------------------------------------------------
    def _toggle_mode(self, *_):
        global MEAS_MODE
        MEAS_MODE = "ACC" if MEAS_MODE == "VEL" else "VEL"
        self.btn_mode.text = f"Mode: {MEAS_MODE}"
        self.log(f"▶ 측정 모드 → {MEAS_MODE}")
    def _save_baseline(self, *_):
        if self.last_fn is None:
            self.log("Fₙ 값 없음")
        else:
            self.F0 = self.last_fn
            self.log(f"F₀ = {self.F0:.2f} Hz 저장")
    # ------------------------------------------------------------------
    def _has_allfiles_perm(self):
        MANAGE = getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)
        return not MANAGE or check_permission(MANAGE)
    def _goto_allfiles_permission(self):
        from jnius import autoclass
        Intent, Settings, Uri = map(autoclass,
            ("android.content.Intent",
             "android.provider.Settings",
             "android.net.Uri"))
        act = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(
            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))
# ------------------------------------------------------------------
if __name__ == "__main__":
    FFTApp().run()
