###############################################################################
# 0. Config ― 반드시 Kivy import *이전*에!
###############################################################################
import os, tempfile, pathlib
from kivy.config import Config

_LOG_DIR = os.path.join(tempfile.gettempdir(), "fftlogs")
os.makedirs(_LOG_DIR, exist_ok=True)
Config.set("kivy", "log_dir",  _LOG_DIR)
Config.set("kivy", "log_level", "debug")

###############################################################################
# 1. 공통 모듈
###############################################################################
import io, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time, re
from collections import deque
import numpy as np
from numpy.fft import fft
import faulthandler, signal, subprocess

###############################################################################
# 2. faulthandler – 디바이스·퍼미션 관계없이 안전하게
###############################################################################

SDCARD_LOG = pathlib.Path("/sdcard/fft_crash.log")
DESKTOP_LOG = pathlib.Path(tempfile.gettempdir()) / "fft_crash.log"

def _get_log_path() -> pathlib.Path:
    """안드로이드면 /sdcard, 아니면 임시폴더."""
    if platform == "android":
        return SDCARD_LOG
    return DESKTOP_LOG            # 윈도·리눅스·맥

def _ensure_parent(p: pathlib.Path):
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _open_log(mode="a", buffering=1):
    p = _get_log_path()
    _ensure_parent(p)
    try:
        return open(p, mode, buffering=buffering, encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        # 최후의 fallback → 메모리 버퍼(로그는 화면에만 남음)
        import io
        return io.StringIO()
   
###############################################################################
# 3. Kivy 및 Android 전용 모듈
###############################################################################
from kivy.utils import platform
from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.spinner import Spinner
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.graphics import Line, Color, PushMatrix, PopMatrix, Translate, Rotate
from plyer import filechooser, accelerometer

# ──────────────────────────────────────────────────────────────────
# Android 전용 준비
# ──────────────────────────────────────────────────────────────────
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
    try:  # Downloads 절대경로 확보
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
        Environment = autoclass("android.os.Environment")
        DOWNLOAD_DIR = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() or "/sdcard/Download"
    except Exception:
        DOWNLOAD_DIR = "/sdcard/Download"
else:
    DOWNLOAD_DIR = os.path.expanduser("~/Download")

###############################################################################
# 4. 사용자 조정 상수
###############################################################################
BAND_HZ            = 0.5
REF_MM_S, REF_ACC  = 0.01, 0.981
MEAS_MODE          = "VEL"          # "VEL" 또는 "ACC"
SMOOTH_N           = 1
HPF_CUTOFF         = 5.0
MAX_FMAX           = 50
REC_DURATION_DEF   = 60.0
FN_BAND            = (5, 50)
BUF_LEN, MIN_LEN   = 4096, 1024      # 실시간 버퍼
USE_SPLIT          = True            # 그래프 3-way 분할
F_MIN = 5



###############################################################################
# 5. 로깅/크래시 헬퍼
###############################################################################
def _dump_crash(txt: str):
    fp = _open_log()
    try:
        fp.write("\n" + "="*60 + "\n" +
                 datetime.datetime.now().isoformat() + "\n" + txt + "\n")
        fp.flush()
    except Exception:
        pass
    Logger.error(txt)


def dump_logcat(tag="fft_logcat"):
    """최근 200줄 logcat → /sdcard/tag_yyyymmdd_HHMMSS.txt"""
    if platform != 'android':
        return
    if not ANDROID:
        return
    try:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"/sdcard/{tag}_{ts}.txt"
        with open(path, "w") as fp:
            subprocess.run(["logcat", "-d", "-t", "200"],
                           stdout=fp, stderr=subprocess.DEVNULL, check=False)
    except Exception as e:
        Logger.warning(f"logcat dump fail: {e}")

def _ex(et, ev, tb):
    dump_logcat("crash")
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)),
                  size_hint=(.9, .9)).open())
sys.excepthook = _ex

###############################################################################
# 6. 공용 함수
###############################################################################
from typing import Optional
def uri_to_file(uri: str) -> Optional[str]:
    if not uri:
        return None
    if uri.startswith("file://"):
        real = urllib.parse.unquote(uri[7:])
        return real if os.path.exists(real) else None
    if uri.startswith("content://") and ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                uri, uuid.uuid4().hex + ".csv", to_downloads=True)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
            return None
    if os.path.exists(uri):
        return uri
    Logger.warning(f"uri_to_file: cannot access {uri}")
    return "NO_PERMISSION"

def acc_to_spec(freq, amp_a):
    if MEAS_MODE == "VEL":
        f_nz = np.where(freq < 1e-6, 1e-6, freq)
        amp  = amp_a / (2 * np.pi * f_nz) * 1e3
        ref  = REF_MM_S
    else:
        amp, ref = amp_a, REF_ACC
    return amp, ref

def smooth_y(vals, n=SMOOTH_N):
    if n <= 1 or len(vals) < n:
        return vals[:]
    return np.convolve(vals, np.ones(n) / n, mode="same")

def welch_band_stats(sig, fs, f_lo=HPF_CUTOFF, f_hi=MAX_FMAX,
                     band_w=BAND_HZ, seg_n=None, overlap=0.5):
    seg_n = seg_n or int(fs * 4)
    step  = int(seg_n * (1 - overlap))
    win   = np.hanning(seg_n)
    spec_sum, ptr = None, 0
    while ptr + seg_n <= len(sig):
        seg = (sig[ptr:ptr+seg_n] - sig[ptr:ptr+seg_n].mean()) * win
        ps  = (abs(np.fft.rfft(seg))**2) / (np.sum(win**2) * fs)
        spec_sum = ps if spec_sum is None else spec_sum + ps
        ptr += step
    if spec_sum is None:
        return [], []
    psd  = spec_sum / ((ptr - step)//step + 1)
    freq = np.fft.rfftfreq(seg_n, d=1/fs)
    sel  = (freq >= f_lo) & (freq <= f_hi)
    freq, psd = freq[sel], psd[sel]
    amp_lin, REF0 = acc_to_spec(freq, np.sqrt(psd*2))

    band_rms, band_pk = [], []
    for lo in np.arange(f_lo, f_hi, band_w):
        hi = lo + band_w
        s  = (freq >= lo) & (freq < hi)
        if not s.any():
            continue
        rms = np.sqrt(np.mean(amp_lin[s]**2))
        pk  = amp_lin[s].max()
        cen = (lo + hi) / 2
        band_rms.append((cen, 20*np.log10(max(rms, REF0*1e-4)/REF0)))
        band_pk.append((cen, 20*np.log10(max(pk,  REF0*1e-4)/REF0)))
    if len(band_rms) >= SMOOTH_N:
        ys = smooth_y([y for _, y in band_rms])
        band_rms = list(zip([x for x, _ in band_rms], ys))
    return band_rms, band_pk

###############################################################################
# 7. 그래프 위젯
###############################################################################
class GraphWidget(Widget):
    PAD_X, PAD_Y, LINE_W = 80, 50, 2.5
    X_TICKS = [5,10,20,30,40,50]
    PEAK_N = 3
    PEAK_MIN_SEP = 2.0
 
    # 축 고유 색상:  X=Red, Y=Green, Z=Blue
    AXIS_CLR = {"x":(1,0,0), "y":(0,1,0), "z":(0,1,1)}
    #   (기존 COLORS 배열은 더 이상 쓰지 않으므로 삭제)
    DIFF_CLR = (1,1,1)

    def __init__(self, **kw):
        super().__init__(**kw)          
        self.datasets, self.diff = [],[]
        self.min_x = F_MIN
        self.max_x = 50.0
        self.X_TICKS = [5,10,20,30,40,50]
        self.Y_MIN, self.Y_MAX = 0, 100
        self.Y_TICKS = [0,20,40,60,80,100]
        self._prev_ticks = (None, None)
        self.bind(size=lambda *_: self.redraw())
        self.status_text = None

       # ───────────────────────────── 축 라벨
        # ── 축 제목(Label) ────────────────────────────────
        self.lbl_x = Label(text="Frequency (Hz)",
                           size_hint=(None, None))
        self.lbl_y = Label(text="Level (dB)",
                           size_hint=(None, None))
        # y축은 Canvas 회전을 써서 90° 세로 배치
        with self.lbl_y.canvas.before:
            PushMatrix()
            Rotate (angle=90, origin=self.lbl_y.center)
        with self.lbl_y.canvas.after:
            PopMatrix()
   
        self.add_widget(self.lbl_x)
        self.add_widget(self.lbl_y)
        # --------------------------------------------------

        self.status_lbl = Label(bold=True,
                                font_size='16sp',
                                color=(0, 1, 0, 1),   # default green
                                size_hint=(None, None))
        self.add_widget(self.status_lbl)

        # 크기 바뀔 때마다 배지·축제목 위치 갱신
        self.bind(size=self._reposition_titles, pos=self._reposition_titles)



    def _peak_label_pos(self, order):
        """
        그래프 오른쪽 위부터 차례로 아래로 내려가며 label 배치.
        order : 0,1,2…  (피크 순서)
        """
        x = self.width - self.PAD_X + 10          # PAD 영역 밖으로 살짝
        y = self.height - self.PAD_Y - 18 - order*18
        return self.x + x, self.y + y             # 부모(BoxLayout) 절대좌표


    def _select_peaks(self, rms_line):
        """
        rms_line : [(freq, level_dB), ...]  — 이미 큰→작 순 정렬되어 있어야 함
        return    : 선택된 (freq, level_dB) 목록
        """
        chosen = []
        for f, lv in rms_line:
            if all(abs(f - cf) >= self.PEAK_MIN_SEP for cf, _ in chosen):
                chosen.append((f, lv))
                if len(chosen) >= self.PEAK_N:
                    break
        return chosen
    
        
   
    def _reposition_titles(self, *_):
        # X축 : 위젯 아래쪽 중앙
        self.lbl_x.texture_update()  # width/height 최신화
        self.lbl_x.pos = (self.x + (self.width - self.lbl_x.texture_size[0]) / 2,
                          self.y + 2)        # 살짝 안쪽

        # Y축 : 왼쪽 가운데 (회전돼 있으므로 width/height 바뀜)
        self.lbl_y.texture_update()
        txt_w, txt_h = self.lbl_y.texture_size
        self.lbl_y.pos = (self.x + 2,                       # 살짝 안쪽
                          self.y + (self.height - txt_w) / 2)  # txt_w ↔︎ txt_h 주의!

        # ── 중앙 상단에 상태 배지 ──────────────────────
 
        self.status_lbl.texture_update()
        bx = self.x + (self.width - self.status_lbl.texture_size[0]) * 0.5
        by = self.y + self.height - self.PAD_Y + 6
        self.status_lbl.pos = (bx, by)
        
   
   
    def _add_axis_label(self, txt: str, loc_xy):
        """loc_xy는 위젯 로컬 좌표;  부모(BoxLayout) 좌표계로 보정해서 Label 추가"""
        lx, ly = loc_xy           # <- local x,y   (위젯 좌하단 기준)
        lbl = Label(text=txt,
                    size_hint=(None, None), size=(60, 20),
                    # 부모 기준 절대 좌표 = 위젯 원점(self.x, self.y) + 로컬좌표
                    pos=(self.x + float(lx), self.y + float(ly)))
        lbl._axis = True
        self.add_widget(lbl)
   
   
    def _make_labels(self):
        if self.width < 5:
            return
        # ① 기존 라벨 제거
        for ch in list(self.children):
            if getattr(ch, "_axis", False):
                self.remove_widget(ch)

        # ② X-축 (0~50 Hz, 10 Hz 간격) ― 모든 그래프에서 표시
        for f in (self.X_TICKS):
            if f < self.min_x or f > self.max_x:
                continue
            rel = (f - self.min_x)/(self.max_x-self.min_x)
            x   = self.PAD_X + rel*(self.width-2*self.PAD_X) - 18
            self._add_axis_label(f"{f} Hz", (x, self.PAD_Y-28))

        # ③ Y-축
        for v in self.Y_TICKS:
            y = self.y_pos(v) - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+8):
                self._add_axis_label(str(v), (x, y))

               

    # ───────────────────────────── 외부 API
    def update_graph(self, ds, df, xm, status: str | None = None):
        # ── 3-튜플(rms, pk, axis) 만 저장
        self.datasets = [seq for seq in (ds or []) if seq and len(seq) == 3]
        self.diff     = df or []
        self.max_x    = min(float(xm), 50.0)
        self.status_text = status
    
   
        # ── y 값 모으기 : rms·pk 에서만
        ys = []
        for rms, pk, _ in self.datasets:
            ys.extend([y for _, y in rms])
            ys.extend([y for _, y in pk])
        ys.extend([y for _, y in self.diff])
   
        # ① y-범위를 20 dB 단위로 '스냅'      (top, low 계산 부분만 바꾸면 끝!)
        if ys:
            raw_top = max(ys)
            raw_low = min(ys)
            top = ((int(raw_top)+19)//20)*20      # 0,20,40,… 로 올림
            low = ((int(raw_low)-19)//20)*20      # 0,-20,-40,… 로 내림
            if (top, low) != (self.Y_MAX, self.Y_MIN):   # '진짜' 바뀔 때만
                self.Y_MIN, self.Y_MAX = low, top
                self.Y_TICKS = list(range(low, top+1, 20))


        # ── 상태 배지 텍스트/색상 ───────────────────────
    
        if status is not None:
            self.status_lbl.text  = status
            self.status_lbl.color = (1,0,0,1) if status.startswith("PLZ") else (0,1,0,1)
        else:
            # 실시간 호출 등으로 배지를 지우고 싶을 때는 None/""를 넘겨서 빈 문자열 처리
            self.status_lbl.text = ""

        # -------- 모든 내부 상태 정리 끝 → 실제 그리기 ----------
        self.redraw(


    # ───────────────────────────── 내부 헬퍼
    def y_pos(self, v):
        h = self.height - 2*self.PAD_Y
        return self.PAD_Y + (v-self.Y_MIN)/(self.Y_MAX-self.Y_MIN) * h

    def _scale(self, pts):
        if not pts or self.width < 5:
            return []
        w = max(self.width-2*self.PAD_X, 1)
        out=[]
        for x,y in pts:
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            sx = float(self.PAD_X + (x-self.min_x)/(self.max_x-self.min_x)*w)
            sy = float(self.y_pos(y))
            out.extend((sx,sy))
        return out

    def _grid(self):
        for f in self.X_TICKS:
            if f < self.min_x or f > self.max_x:
                continue
            rel = (f - self.min_x)/(self.max_x-self.min_x)
            gx  = self.PAD_X + rel*(self.width-2*self.PAD_X)
            Line(points=[gx, self.PAD_Y,
                         gx, self.height-self.PAD_Y])

        for v in self.Y_TICKS:
            Line(points=[self.PAD_X, self.y_pos(v),
                         self.width-self.PAD_X, self.y_pos(v)])

    def _safe_line(self, pts, dash=False):
        if len(pts) < 4:
            return
        MAX_V = 4094
        if dash:
            dash_len, gap_len = 1.5, 4.
            for i in range(0,len(pts)-2,2):
                x1,y1,x2,y2 = pts[i:i+4]
                seg = ((x2-x1)**2+(y2-y1)**2)**0.5
                if seg < 1e-6:
                    continue
                nx,ny,ofs,draw = (x2-x1)/seg,(y2-y1)/seg,0.,True
                while ofs < seg:
                    ln = min(dash_len if draw else gap_len, seg-ofs)
                    if draw:
                        self._safe_line([x1+nx*ofs,y1+ny*ofs,
                                         x1+nx*(ofs+ln),y1+ny*(ofs+ln)])
                    ofs += ln; draw = not draw
            return
        for i in range(0, len(pts), MAX_V):
            chunk = pts[i:i+MAX_V]
            if len(chunk) >= 4:
                Line(points=chunk, width=self.LINE_W)

    # ───────────────────────────── 핵심 redraw
    def redraw(self, *_):
        self.canvas.clear()
        for ch in list(self.children):
            if getattr(ch, "_axis", False) or getattr(ch, "_peak", False):
                self.remove_widget(ch)
   
        if not self.datasets and len(self.diff)< 2:
            return
   
        if (self.max_x, (self.Y_MIN, self.Y_MAX)) != self._prev_ticks:
            self._make_labels()
            self._prev_ticks = (self.max_x, (self.Y_MIN, self.Y_MAX))


        with self.canvas:
            PushMatrix()
            Translate(self.x, self.y)

            self._grid()
            peaks = []                         # (sx, sy, fx, axis, order)

            for rms, pk, axis in self.datasets:
                # ① 스펙트럼 선 그리기 ------------------------------------
                for j, pts in enumerate((rms, pk)):
                    if len(pts) < 2:
                        continue
                    Color(*self.AXIS_CLR[axis])
                    self._safe_line(self._scale(pts), dash=bool(j))

                # ② 이 축에서 큰 순으로 PEAK_N 개 좌표 수집 ---------------
                # ① 기존 for-루프 안에서 rms 최대값을 찾는 부분을 ↓처럼 바꿉니다
                sorted_rms = sorted(rms, key=lambda p: p[1], reverse=True)
                for k, (fx, fy) in enumerate(self._select_peaks(sorted_rms)):
                    sx, sy = self._scale([(fx, fy)])[:2]
                    peaks.append((sx, sy, fx, axis, k))

            # diff 그래프 (흰색) --------------------------------------------
            if len(self.diff) >= 2:
                Color(1, 1, 1)
                self._safe_line(self._scale(self.diff))

            PopMatrix()

            # ── ★ 배지 그리기 -------------------------------
            if self.status_text:
                badge_w, badge_h = 90, 32
                x0 = self.width - badge_w - 10
                y0 = self.height - badge_h - 10
                # 색상 : GOOD→초록 , PLZ CHECK→빨강
                if self.status_text == "GOOD":
                    Color(0, .7, 0, .8)
                else:                    # "PLZ CHECK"
                    Color(.9, 0, 0, .8)
                # 배경 사각형
                Line(points=[x0, y0, x0+badge_w, y0,
                             x0+badge_w, y0+badge_h, x0, y0+badge_h,
                             x0, y0], width=0)  # 닫힌 폴리라인
                # 텍스트
                lbl = Label(text=self.status_text,
                            size_hint=(None, None),
                            size=(badge_w, badge_h),
                            pos=(self.x + x0, self.y + y0))
                lbl._badge = True
                self.add_widget(lbl)


        # ── ③ 피크 라벨 배치 -----------------------------------------------
        for sx, sy, fx, axis, order in peaks:
            lbl = Label(text=f"{fx:.1f} Hz",
                        color=self.AXIS_CLR[axis] + (1,),
                        size_hint=(None, None))
            lbl.texture_update()
            w, h = lbl.texture_size

            # — 꼭짓점 위 조금 띄워서, 그래프 영역 안에 들어오도록 조정 —
            px = self.x + sx - w / 2
            py = self.y + min(sy + 8 + order * 14,   # 겹치면 위로 조금씩 밀기
                              self.height - h - 4)

            lbl.pos = (px, py)
            lbl._peak = True
            self.add_widget(lbl)

        # 기존 피크 라벨 처리 아래에 ↓ (배지 Label 제거용)
        for ch in list(self.children):
            if getattr(ch, "_badge", False) and ch.text != self.status_text:
                self.remove_widget(ch)

        
###############################################################################
# 8. 메인 앱
###############################################################################
class FFTApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.rt_on = self.rec_on = False
        self.rt_buf = {ax: deque(maxlen=BUF_LEN) for ax in "xyz"}
        self.rec_start = 0.0
        self.rec_files = {}
        self.REC_DURATION = REC_DURATION_DEF
        self.last_fn = self.F0 = None
        self._buf_lock = threading.Lock()


    # ───────────────────────────── UI
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # 안내
        self.label = Label(text="Pick up to 2 CSV", size_hint=(1,.05))
        root.add_widget(self.label)

        # 버튼
        self.btn_sel = Button(text="Select CSV", size_hint=(1,.05),
                              disabled=True, on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",  size_hint=(1,.05),
                              disabled=True, on_press=self.run_fft)
        self.btn_rec = Button(text=f"Record {int(self.REC_DURATION)} s",
                              size_hint=(1,.05), disabled=True,
                              on_press=self.start_recording)
        root.add_widget(self.btn_sel), root.add_widget(self.btn_run), root.add_widget(self.btn_rec)

        # 스피너
        self.spin_dur = Spinner(text=f"{int(self.REC_DURATION)} s",
                                values=("10 s","30 s","60 s","120 s"),
                                size_hint=(1,.05))
        self.spin_dur.bind(text=lambda s,t: self._set_rec_dur(float(t.split()[0])))
        self.spin_sm  = Spinner(text=str(SMOOTH_N),
                                values=("1","2","3","4","5"),
                                size_hint=(1,.05))
        self.spin_sm.bind(text=lambda s,t: self._set_smooth(int(t)))
        root.add_widget(self.spin_dur), root.add_widget(self.spin_sm)

        # 모드/Realtime
        self.btn_mode = Button(text=f"Mode: {MEAS_MODE}", size_hint=(1,.05),
                               on_press=self._toggle_mode)
        self.btn_setF0 = Button(text="Set F0 (baseline)", size_hint=(1,.05),
                                on_press=self._save_baseline)
        self.btn_rt = Button(text="Realtime FFT (OFF)", size_hint=(1,.05),
                             on_press=self.toggle_realtime)
        root.add_widget(self.btn_mode), root.add_widget(self.btn_setF0), root.add_widget(self.btn_rt)

        # 그래프 3-way
        self.graphs=[]
        gbox = BoxLayout(orientation="vertical", size_hint=(1,.60), spacing=4)
        for _ in range(3):
            gw = GraphWidget(size_hint=(1,1/3))
            self.graphs.append(gw)
            gbox.add_widget(gw)
        root.add_widget(gbox)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ───────────────────────────── 헬퍼
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        if ANDROID and toast:
            try:
                toast.toast(msg)
            except Exception:
                pass

    def _set_rec_dur(self, sec):
        self.REC_DURATION = sec
        self.btn_rec.text = f"Record {int(sec)} s"

    def _set_smooth(self, n):
        global SMOOTH_N
        SMOOTH_N = n
        self.log(f"▶ SMOOTHING WINDOW = {SMOOTH_N}")

    # ───────────────────────────── 권한
    def _ask_perm(self, *_):
        if not ANDROID or SharedStorage:        # 데스크톱/SAF picker
            self.btn_sel.disabled = self.btn_rec.disabled = False
            return
        need = [getattr(Permission,"READ_EXTERNAL_STORAGE",""),
                getattr(Permission,"WRITE_EXTERNAL_STORAGE","")]
        MANAGE = getattr(Permission,"MANAGE_EXTERNAL_STORAGE",None)
        if MANAGE:
            need.append(MANAGE)
        if ANDROID_API >= 33:
            need += [getattr(Permission,n,"") for n in
                     ("READ_MEDIA_IMAGES","READ_MEDIA_AUDIO","READ_MEDIA_VIDEO")]
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = self.btn_rec.disabled = False
        else:
            request_permissions(need, lambda *_:
                (setattr(self.btn_sel,"disabled",False),
                 setattr(self.btn_rec,"disabled",False)))

    # ───────────────────────────── 레코딩
    def start_recording(self, *_):
        if self.rec_on:
            self.log("RECORDING..."); return
        try:
            accelerometer.enable()
        except Exception as e:
            self.log(f"DO NOT USE THE SENSOR: {e}"); return
        if not self.rt_on:     # 실시간 FFT 동시 시작
            self.toggle_realtime()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_files = {}
        try:
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            for ax in "xyz":
                fp = open(os.path.join(DOWNLOAD_DIR,f"acc_{ax}_{ts}.csv"),
                          "w", newline="", encoding="utf-8")
                csv.writer(fp).writerow(("time","acc"))
                self.rec_files[ax] = fp
        except Exception as e:
            self.log(f"FAILED OPEN FILE: {e}")
            return
        self.rec_on = True
        self.rec_start = time.time()
        self.btn_rec.disabled = True
        self.label.text = f"Recording 0/{int(self.REC_DURATION)} s …"
        Clock.schedule_once(self._stop_recording, self.REC_DURATION)

    def _stop_recording(self, *_):
        if not self.rec_on:
            return
        for fp in self.rec_files.values():
            try: fp.close()
            except Exception: pass
        self.rec_files.clear()
        self.rec_on = False
        self.btn_rec.disabled = False
        self.log("Recording complete!")

    # ───────────────────────────── Realtime poll
    def toggle_realtime(self, *_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
            except Exception as e:
                self.log(f"DO NOT USE THE SENSOR: {e}")
                self.rt_on=False
                self.btn_rt.text="Realtime FFT (OFF)"; return
            Clock.schedule_interval(self._poll_accel, 0)
            threading.Thread(target=self._rt_fft_loop, daemon=True).start()
        else:
            try:
                accelerometer.disable()
            except Exception:
                pass

    def _poll_accel(self, dt):
        if not self.rt_on:
            return False
        try:
            ax,ay,az = accelerometer.acceleration
            if None in (ax,ay,az):
                return
            now = time.time()
            with self._buf_lock:
                for axis,val in zip("xyz",(ax,ay,az)):
                    prev = self.rt_buf[axis][-1][0] if self.rt_buf[axis] else now-dt
                    self.rt_buf[axis].append((now,val,now-prev))

                
            if self.rec_on:
                rel = now-self.rec_start
                for a,v in zip("xyz",(ax,ay,az)):
                    csv.writer(self.rec_files[a]).writerow((rel,v))
                if int(rel*2)%2==0:
                    self.label.text=f"Recording {rel:4.1f}/{int(self.REC_DURATION)} s …"
        except Exception as e:
            Logger.warning(f"acc read fail: {e}")

    # ───────────────────────────── Realtime FFT 루프
    def _rt_fft_loop(self):
        """백그라운드 스레드 – 0.5 s마다 버퍼 스냅샷 → Welch 분석 → 그래프 갱신"""
        FFT_LEN_SEC    = 8
        RT_REFRESH_SEC = 0.5
        MIN_FS         = 50

        try:
            while self.rt_on:
                time.sleep(RT_REFRESH_SEC)

                # ① 버퍼 스냅샷 -------------------------------------------------
                with self._buf_lock:
                    snap = {ax: list(self.rt_buf[ax]) for ax in "xyz"}


                axis_sets, xmax = {}, 0.0                     # ★ 변경 ①

                # ② 축별 FFT ----------------------------------------------------
                for ax in "xyz":
                    if len(snap[ax]) < MIN_FS * FFT_LEN_SEC:
                        continue
                    *_, dt_arr = zip(*snap[ax])
                    dt_arr = np.array(dt_arr[-512:])
                    dt_arr = dt_arr[dt_arr > 1e-5]
                    if not dt_arr.size:
                        continue
                    fs = 1.0 / np.median(dt_arr)
                    if fs < MIN_FS:
                        continue

                    fft_len = int(fs * FFT_LEN_SEC)
                    if len(snap[ax]) < fft_len:
                        continue

                    _, val, _ = zip(*snap[ax][-fft_len:])
                    sig  = (np.asarray(val) - np.mean(val)) * np.hanning(fft_len)

                    spec  = np.fft.rfft(sig)
                    freq  = np.fft.rfftfreq(fft_len, d=1/fs)
                    amp_a = 2 * np.abs(spec) / (fft_len*np.sqrt(2))
                    amp_lin, ref0 = acc_to_spec(freq, amp_a)

                    rms_line, pk_line = [], []
                    FMAX = min(50, fs*0.5)
                    for lo in np.arange(HPF_CUTOFF, FMAX, BAND_HZ):
                        hi  = lo + BAND_HZ
                        sel = (freq >= lo) & (freq < hi)
                        if not sel.any():
                            continue
                        cen = (lo + hi) * 0.5
                        rms = np.sqrt(np.mean(amp_lin[sel]**2))
                        pk  = amp_lin[sel].max()
                        rms_line.append((cen, 20*np.log10(max(rms, ref0*1e-4)/ref0)))
                        pk_line .append((cen, 20*np.log10(max(pk , ref0*1e-4)/ref0)))

                    if not rms_line:
                        continue

                    # 공진 추적
                    f_centres = np.array([x for x, _ in rms_line])
                    f_vals    = np.array([y for _, y in rms_line])
                    band_sel  = (f_centres >= FN_BAND[0]) & (f_centres <= FN_BAND[1])
                    if band_sel.any():
                        self.last_fn = f_centres[band_sel][f_vals[band_sel].argmax()]


                    axis_sets[ax] = (rms_line, pk_line, ax)    # ★ 변경 ②
                    xmax = max(xmax, FMAX)

                # ③ 메인스레드에 그래프 업데이트 ---------------------------------

                # ── ③ 메인스레드에 그래프 업데이트 ─────────────────────
                if axis_sets:
                    def _update(_dt, sets=axis_sets, xm=xmax):
                        for ax, g in zip("xyz", self.graphs):
                            if ax in sets:                           # 데이터 있는 축
                                g.update_graph([sets[ax]], [], xm, "")   # ← status=""
                            else:                                    # 샘플 부족한 축
                                g.update_graph([],        [], xm, "")    # ← status=""
                    Clock.schedule_once(_update)        # ★ 변경 ④

        except Exception:
            Logger.exception("Realtime FFT thread crashed")
            self.rt_on = False
            Clock.schedule_once(lambda *_:
                setattr(self.btn_rt, "text", "Realtime FFT (OFF)"))
    
    # ───────────────────────────── CSV FFT (백그라운드)
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()


  # ────────────────────────────────────────────
    #  CSV 2 개 FFT → ①RMS/Peak 2 세트 ②차이(dB)
    # ────────────────────────────────────────────
    def _fft_bg(self):
        try:
            if len(self.paths) < 2:
                raise ValueError("SELECT 2 CSV FILE")

            data, xmax = [], 0.0                         # ← 결과 보관
            for path in self.paths[:2]:                  # 첫 두 파일만
                t, a = self._load_csv(path)
                if t is None:
                    raise ValueError(f"{os.path.basename(path)}: CSV parse Failed")

                # ── FFT 스펙트럼 ───────────────────────────────────
                dt_arr = np.diff(t);     dt_arr = dt_arr[dt_arr > 0]
                if not dt_arr.size:
                    raise ValueError("non-positive dt in CSV")
                dt   = float(np.median(dt_arr))
                nyq  = 0.5 / dt
                FMAX = max(HPF_CUTOFF + BAND_HZ, min(nyq, MAX_FMAX))

                win  = np.hanning(len(a))
                raw  = np.fft.fft((a - a.mean()) * win)
                amp_a = 2 * np.abs(raw[:len(a)//2]) / (len(a)*np.sqrt(2))
                freq  = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]

                sel   = (freq >= HPF_CUTOFF) & (freq <= FMAX)
                freq, amp_a = freq[sel], amp_a[sel]
                amp_lin, REF0 = acc_to_spec(freq, amp_a)

                # ── 0.5 Hz 밴드 RMS·Peak(dB) ─────────────────────
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

                if len(rms_line) >= SMOOTH_N:            # 스무딩
                    rms_line = list(zip(
                        [x for x, _ in rms_line],
                        smooth_y([y for _, y in rms_line])))

                data.append((rms_line, pk_line))         # ① 저장
                xmax = max(xmax, FMAX)

            # ── ② RMS 차이(dB) 구하기 ────────────────────────────
            f1 = dict(data[0][0]);  f2 = dict(data[1][0])
            diff_line = [(f, f1[f] - f2[f]) for f in sorted(set(f1) & set(f2))]

           
            # ── NEW : 15 dB 이상 차이가 있나?  --------------------
            LIMIT_DB = 15
            alert_msg = "PLZ CHECK" if any(abs(d) >= LIMIT_DB for _, d in diff_line) else "GOOD"

 
            
            # ── 메인-스레드 업데이트 ──────────────────────────────
            def _update(_dt):
                rms0, pk0 = data[0]
                rms1, pk1 = data[1]
    
                # 그래프 0,1 : 원본 스펙 (status 기본값 "")
                self.graphs[0].update_graph([(rms0, pk0, 'x')], [], xmax)
                self.graphs[1].update_graph([(rms1, pk1, 'y')], [], xmax)
    
                # 그래프 2 : diff  +  상태 배지
                self.graphs[2].update_graph([], diff_line, xmax, alert_msg)
    
                # 화면 하단/토스트 메시지도 동일하게
                self.log(alert_msg)
    
            Clock.schedule_once(_update)
    
        except Exception as exc:
            Clock.schedule_once(lambda _dt: self.log(f"FFT 오류: {exc}"))
        finally:
            Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))

           

    # ───────────────────────────── CSV 로드
    def _load_csv(self, path):
        num_re = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?$")
        try:
            t,a = [],[]
            with open(path, encoding="utf-8", errors="replace") as f:
                sample=f.read(1024); f.seek(0)
                try:
                    dialect=csv.Sniffer().sniff(sample, delimiters=";, \t")
                except csv.Error:
                    dialect=csv.get_dialect("excel")
                for row in csv.reader(f,dialect):
                    if len(row)<2: continue
                    if not(num_re.match(row[0].strip())
                           and num_re.match(row[1].strip())): continue
                    t.append(float(row[0].replace(",",".")))
                    a.append(float(row[1].replace(",",".")))
            return (None,None) if len(a)<2 else (np.asarray(t,float), np.asarray(a,float))
        except Exception as e:
            Logger.error(f"CSV read err: {e}")
            return None,None

    # ───────────────────────────── 기타 버튼
    def _toggle_mode(self, *_):
        global MEAS_MODE
        MEAS_MODE = "ACC" if MEAS_MODE == "VEL" else "VEL"
        self.btn_mode.text = f"Mode: {MEAS_MODE}"
        self.log(f"MEASURING MODE : {MEAS_MODE}")

    def _save_baseline(self, *_):
        if self.last_fn is None:
            self.log("Fₙ NO DATA")
        else:
            self.F0 = self.last_fn
            self.log(f"F0 = {self.F0:.2f} Hz SAVED")

    # ───────────────────────────── SAF & 권한
    def _has_allfiles_perm(self):
        MANAGE = getattr(Permission,"MANAGE_EXTERNAL_STORAGE",None)
        return not MANAGE or check_permission(MANAGE)

    def _goto_allfiles_permission(self):
        from jnius import autoclass
        Intent,Settings,Uri = map(autoclass,
            ("android.content.Intent","android.provider.Settings","android.net.Uri"))
        act = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))

    # ───────────────────────────── 파일 선택
       
    def open_chooser(self, *_):
        # ---------- 1) Android : SAF 또는 plyer ----------
        if ANDROID:
            if SharedStorage:
                try:
                    SharedStorage().open_file(callback=self.on_choose,
                                              multiple=True, mime_type="text/*")
                    return
                except Exception:
                    pass
            # plyer fallback
            filechooser.open_file(on_selection=self.on_choose,
                                  multiple=True, filters=["*.csv"],
                                  path=DOWNLOAD_DIR)
            return
   
        # ---------- 2) Windows / macOS / Linux ----------
        try:
            # Tkinter는 파이썬 표준 라이브러리라 별도 설치 불필요
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()        # 창 감춤
            paths = filedialog.askopenfilenames(
                title="Select up to 2 CSV",
                filetypes=[("CSV files", "*.csv")])
            # Tk 객체 해제
            root.update(); root.destroy()
            self.on_choose(list(paths)[:3])
        except Exception as e:
            # 그래도 실패하면 Kivy FileChooserPopup으로 최종 fallback
            Logger.warning(f"Tk file dialog error: {e}")
            filechooser.open_file(on_selection=self.on_choose,
                                  multiple=True, filters=["*.csv"])

    def on_choose(self, sel, *_):
        if not sel:
            return
        self.paths = []
        for raw in sel[:2]:  # ★ 최대 2개까지만 허용
            real = uri_to_file(raw)
            if real == "NO_PERMISSION":
                self.log("NO PERMISSION – TRY SAF Picke"); return
            if not real:
                self.log("FAILED COPY"); return
            self.paths.append(real)
        self.label.text = " · ".join(os.path.basename(p) for p in self.paths)
        self.btn_run.disabled = False

###############################################################################
# 9. Run
###############################################################################
if __name__ == "__main__":
    FFTApp().run()
 
