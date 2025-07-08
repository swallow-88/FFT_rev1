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
USE_SPLIT = True


# ── 파일 맨 위 가까이에 추가 ───────────────────────────────

import io, faulthandler, signal, os, tempfile

def _safe_faulthandler():
    """
    /sdcard 가 막혀 있으면 앱 전용 디렉터리로 자동 fallback.
    파일 열기에 실패해도 StringIO 로 대체해 faulthandler 가
    enable() 단계에서 죽지 않도록 한다.
    """
    paths = ["/sdcard/fft_crash.log",
             os.path.join(tempfile.gettempdir(), "fft_crash.log")]
    for p in paths:
        try:
            fp = open(p, "a", buffering=1)
            break
        except PermissionError:
            fp = None
    if fp is None:                       # 모두 실패 ⇒ 메모리 버퍼라도
        fp = io.StringIO()
    faulthandler.enable(file=fp, all_threads=True)
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGQUIT):
        faulthandler.register(sig, file=fp, all_threads=True)

_safe_faulthandler()

from kivy.config import Config
Config.set('kivy', 'log_level', 'debug')
Config.set('kivy', 'log_enable', '1')
Config.set('kivy', 'log_dir',  '/sdcard')            # 폴더 바꿀 수 있음
Config.set('kivy', 'log_name', 'fft_kivy_%y-%m-%d_%_.txt')
Config.write()

import subprocess, os, time

def dump_logcat(tag="fft_logcat"):
    """최근 200줄 logcat 을 /sdcard/tag_yyyyMMdd_HHmmss.txt 로 저장"""
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"/sdcard/{tag}_{ts}.txt"
        with open(path, "w") as fp:
            subprocess.run(["logcat", "-d", "-t", "200"], stdout=fp, check=False)
    except Exception as e:
        Logger.warning(f"logcat dump fail: {e}")


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
    dump_logcat("crash")                 # ★ logcat 스냅샷
    txt = "".join(traceback.format_exception(et, ev, tb))
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
        self._prev_ticks = (None, None)
        self.bind(size=self.redraw)

 
    def _make_labels(self):
        """X·Y 축 라벨을 새로 만듦 (tick 변경 시에만 호출)"""
        if self.width < 5:
            return
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)
        # X 축
        n_tick = int(self.max_x // 10) + 1
        span   = max(n_tick - 1, 1)
        for i in range(n_tick):
            x = self.PAD_X + i * (self.width - 2*self.PAD_X) / span - 18
            lbl = Label(text=f"{10*i} Hz", size_hint=(None,None), size=(60,20),
                        pos=(x, self.PAD_Y-28)); lbl._axis = True
            self.add_widget(lbl)
        # Y 축
        for v in self.Y_TICKS:
            y = self.y_pos(v) - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+8):
                lbl = Label(text=f"{v}", size_hint=(None,None), size=(60,20),
                            pos=(x, y)); lbl._axis = True
                self.add_widget(lbl)



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
        """(freq, dB) → [x1,y1,x2,y2,…]  (NaN 필터·짝수 길이 보장)"""
        if not pts or self.width < 5:                               # ★
            return []
        w = max(self.width - 2 * self.PAD_X, 1)
        out = []
        for x, y in pts:
            if not np.isfinite(x) or not np.isfinite(y):            # ★
                continue
            sx = self.PAD_X + (x / max(self.max_x, 1e-6)) * w
            sy = self.y_pos(y)
            out.extend((sx, sy))
        return out

    # ..............................................
    def _grid(self):
        n_tick = int(self.max_x // 10) + 1
        if n_tick > 80:                           # ★ 80개 이상이면 간격 늘리기
            step_hz = 10 * ((n_tick // 80) + 1)
            n_tick  = int(self.max_x // step_hz) + 1
        gx = (self.width - 2*self.PAD_X) / max(n_tick - 1, 1)

     
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
    # ★ 1) 안전한 dashed_line – 0-length·NaN 방지 + 버텍스 분할
    def _safe_line(self, points, dash=False):
        """
        points: 1-D [x1,y1,x2,y2,…]  (len >= 4, 짝수)
        dash  : True ⇒ 점선 (고정 패턴)
        """
        MAX_VERT = 4094        # Mali 일부 칩셋에서 안전한 한계
        if dash:
            dash_len, gap_len = 10.0, 6.0
            for i in range(0, len(points)-2, 2):
                x1, y1, x2, y2 = points[i:i+4]
                seg_len = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                if seg_len < 1e-9:                          # ★ 0-길이 skip
                    continue
                nx, ny, s, draw = (x2-x1)/seg_len, (y2-y1)/seg_len, 0.0, True
                while s < seg_len:
                    l = min(dash_len if draw else gap_len, seg_len - s)
                    if draw:
                        self._safe_line([x1+nx*s, y1+ny*s,
                                         x1+nx*(s+l), y1+ny*(s+l)], dash=False)
                    s, draw = s + l, not draw
            return

        # ★ Line() 당 MAX_VERT 초과 시 블록 단위로 분할
        for i in range(0, len(points), MAX_VERT):
            seg = points[i:i+MAX_VERT]
            if len(seg) >= 4:
                Line(points=seg, width=self.LINE_W)

    # ★ 2) redraw() – _safe_line 호출로 교체
    def redraw(self, *_):
        self.canvas.clear()
        self._clear_labels()                 
	# 화면에 그리기 전에 label 정리

	# ── 피크 라벨               # ★ ④ 먼저 기존 지우기
        cur_ticks = (self.max_x, (self.Y_MIN, self.Y_MAX))
        if cur_ticks != self._prev_ticks:    # 새 tick이면 만들기
            self._make_labels()
            self._prev_ticks = cur_ticks

        with self.canvas:
            self._grid()
            peaks = []
            for idx, pts in enumerate(self.datasets):
                if len(pts) < 2:
                    continue
                Color(*self.COLORS[idx // 2 % len(self.COLORS)])
                scaled = self._scale(pts)
                if len(scaled) < 4:
                    continue

                if idx % 2:         # Peak(점선)
                    self._safe_line(scaled, dash=True)
                else:               # RMS(실선)
                    self._safe_line(scaled, dash=False)
                    fx, fy = max(pts, key=lambda p: p[1])
                    sx, sy = self._scale([(fx, fy)])[0:2]
                    peaks.append((fx, fy, sx, sy))

            if len(self.diff) >= 2:
                Color(*self.DIFF_CLR)
                self._safe_line(self._scale(self.diff), dash=False)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None, None), size=(90, 22),
                        pos=(sx-30, sy+6))
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
        return root          # ← build() 끝, 아래 중복 블록 삭제
 

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
    # ------------------------------------------------------------------
    #  실시간 FFT 루프  (0.5 s 주기, Welch + 0.5 Hz 밴드 RMS·Peak)
    # ------------------------------------------------------------------
    def _rt_fft_loop(self):
        try:
            while self.rt_on:
                time.sleep(0.5)

                # 1) 버퍼 스냅샷 (쓰레드 lock)
                with self._buf_lock:
                    if any(len(self.rt_buf[a]) < MIN_LEN for a in "xyz"):
                        continue                                # 버퍼 부족
                    buf_copy = {a: list(self.rt_buf[a]) for a in "xyz"}

                axis_sets, xmax = {}, 0.0

                # 2) 축별 FFT ------------------------------------------
                for axis in "xyz":
                    ts, val, dt_arr = zip(*buf_copy[axis])

                    # ── 샘플링 주파수 실측 --------------------------
                    dt_seg = np.array(dt_arr[-512:])
                    dt_seg = dt_seg[dt_seg > 1e-5]             # 100 µs 이하·0 제거
                    if dt_seg.size == 0:
                        continue
                    fs = 1.0 / float(np.median(dt_seg))
                    if fs < 2 * (HPF_CUTOFF + BAND_HZ):        # Nyquist < 분석 밴드
                        continue
                    f_hi = min(fs * 0.5, MAX_FMAX)

                    # ── Welch 스펙트럼 → 0.5 Hz 밴드 RMS·Peak ----
                    band_rms, band_pk = welch_band_stats(
                        np.asarray(val, float),
                        fs      = fs,
                        f_lo    = HPF_CUTOFF,
                        f_hi    = f_hi,
                        band_w  = BAND_HZ)

                    if not band_rms:
                        continue

                    axis_sets[axis] = (band_rms, band_pk)
                    xmax = max(xmax, f_hi)

                    # ── 공진수(Fₙ) 실시간 추적 ------------------
                    loF, hiF = FN_BAND
                    freqs = np.array([x for x, _ in band_rms])
                    mags  = np.array([y for _, y in band_rms])
                    s = (freqs >= loF) & (freqs <= hiF)
                    if s.any():
                        self.last_fn = freqs[s][mags[s].argmax()]


                # 3) 그래프 갱신 ---------------------------------------
                if axis_sets:
                    def _update(dt):
                        for idx, axis in enumerate("xyz"):
                            rms, pk = axis_sets.get(axis, ([], []))
                            self.graphs[idx].update_graph([rms, pk], [], xmax)

                    Clock.schedule_once(_update, 0.05)

        except Exception:
            Logger.exception("Realtime FFT thread crashed")
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
            graph_data = {0: ([], []), 1: ([], []), 2: ([], [])}
            xmax = 0.0

            # ── 파일마다 처리 ───────────────────────────────────────────
            for f_idx, path in enumerate(self.paths):
                t, a = self._load_csv(path)
                if t is None:
                    raise ValueError(f"{os.path.basename(path)}: CSV parse fail")

                # ① 샘플링 파라미터 ------------------------------------
                dt_arr = np.diff(t)
                dt_arr = dt_arr[dt_arr > 0]          # 0·음수 제거★
                if dt_arr.size == 0:
                    raise ValueError("non-positive dt in CSV")
                dt   = float(np.median(dt_arr))
                nyq  = 0.5 / dt
                FMAX = min(nyq, MAX_FMAX)
                if FMAX < HPF_CUTOFF + BAND_HZ:      # 최소 한 밴드 확보★
                    FMAX = HPF_CUTOFF + BAND_HZ

                # ② FFT 스펙트럼 ---------------------------------------
                sig   = (a - a.mean()) * np.hanning(len(a))
                raw   = np.fft.fft(sig)
                amp_a = 2 * np.abs(raw[:len(a)//2]) / (len(a)*np.sqrt(2))
                freq  = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]

                sel = (freq >= HPF_CUTOFF) & (freq <= FMAX)
                freq, amp_a = freq[sel], amp_a[sel]
                amp_lin, REF0 = acc_to_spec(freq, amp_a)

                # ③ 0.5 Hz 밴드 RMS·Peak(dB) --------------------------
                rms_line, pk_line = [], []
                for lo in np.arange(HPF_CUTOFF, FMAX, BAND_HZ):
                    hi = lo + BAND_HZ
                    m  = (freq >= lo) & (freq < hi)
                    if not m.any():
                        continue
                    cen = (lo + hi) / 2
                    rms = np.sqrt(np.mean(amp_lin[m]**2))
                    pk  = amp_lin[m].max()
                    rms_line.append((cen, 20*np.log10(max(rms, REF0*1e-4)/REF0)))
                    pk_line .append((cen, 20*np.log10(max(pk , REF0*1e-4)/REF0)))

                if len(rms_line) >= SMOOTH_N:
                    rms_line = list(zip(
                        [x for x, _ in rms_line],
                        smooth_y([y for _, y in rms_line])))

                # 공진수(Fₙ) 추적 ------------------------------★ 들여쓰기 0
                loF, hiF = FN_BAND
                if rms_line:
                    freq_cent = np.array([x for x, _ in rms_line])
                    mag       = np.array([y for _, y in rms_line])
                    s = (freq_cent >= loF) & (freq_cent <= hiF)
                    if s.any():
                        self.last_fn = freq_cent[s][mag[s].argmax()]

                # ⑤ 그래프 축 결정 ------------------------------------
                m   = re.search(r"_([xyz])_", os.path.basename(path).lower())
                idx = {"x":0, "y":1, "z":2}.get(m.group(1)) if m else f_idx % 3
                graph_data[idx] = (rms_line, pk_line)
                xmax = max(xmax, FMAX)

            # ── UI 업데이트 (메인 스레드) ───────────────────────────
            def _update(*_):
                if USE_SPLIT:
                    for i in range(3):
                        rms, pk = graph_data[i]
                        self.graphs[i].update_graph([rms, pk], [], xmax)
                else:
                    ds = []
                    for i in range(3):
                        rms, pk = graph_data[i]
                        ds += rms + pk
                    self.graph.update_graph(ds, [], xmax)
            Clock.schedule_once(_update)

        except Exception as e:
            Clock.schedule_once(lambda *_: self.log(f"FFT 오류: {e}"))

        finally:
            Clock.schedule_once(lambda *_:
                setattr(self.btn_run, "disabled", False))
         
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
