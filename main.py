###############################################################################
# 0. Config ― 반드시 Kivy import *이전*에!
###############################################################################
import os, tempfile, pathlib
from kivy.config import Config

_LOG_DIR = os.path.join(tempfile.gettempdir(), "fftlogs")
os.makedirs(_LOG_DIR, exist_ok=True)
Config.set("kivy", "log_dir",  _LOG_DIR)
Config.set("kivy", "log_level", "debug")

##########################ㅈ#####################################################
# 1. 공통 모듈
###############################################################################
import io, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time, re
from collections import deque
import numpy as np
from numpy.fft import fft
import faulthandler, signal, subprocess

try:
    import scipy.signal as ss        # DPSS, Welch 창
except ImportError:
    ss = None
    USE_MULTITAPER = False           # SciPy 없으면 Welch 로만

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
from kivy.graphics import Line, Color, PushMatrix, PopMatrix, Translate, Rotate, RoundedRectangle
from plyer import filechooser, accelerometer
from kivy.metrics import dp

#from utils_fft import robust_fs, next_pow2


# ------------------------------------------------------------------
# FFT 보조 함수 – utils_fft.py 내용을 그대로 가져옴
# ------------------------------------------------------------------

def robust_fs(t_arr, min_fs=10.0, n_keep=None):
    """
    t_arr : 타임스탬프 배열   (len ≥ 3)
    n_keep: 뒤에서부터 n_keep 개만 사용 (메모리 절약용)
    """
    if n_keep:
        t_arr = t_arr[-n_keep:]
    if len(t_arr) < 3:
        return None
    dt = np.diff(t_arr)
    dt = dt[dt > 1e-6]
    if not dt.size:
        return None
    fs = 1.0 / np.median(dt)
    return fs if fs >= min_fs else None


def next_pow2(n):
    """
    n 이상 가장 작은 2의 거듭제곱 정수 반환.
    예)  500 → 512 , 1024 → 1024
    """
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


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
# ── 파일 맨 위 ‘사용자 조정 상수’ 자리 근처 ──────────────
# ── 4. 사용자 조정 상수 ───────────────────────────
# ── 4. 사용자 조정 상수 ───────────────────────────
CFG = dict(BAND_HZ=0.5, HPF_CUTOFF=5.0, MAX_FMAX=50.0, SMOOTH_N=1)

# ↓ 옛 변수명 유지(호환) ────────────────────────────
HPF_CUTOFF = CFG["HPF_CUTOFF"]      # ← 이 줄 추가
BAND_HZ    = CFG["BAND_HZ"]
SMOOTH_N   = CFG["SMOOTH_N"]
MAX_FMAX   = CFG["MAX_FMAX"]
# --------------------------------------------------

#BAND_HZ            = 0.5
REF_MM_S, REF_ACC  = 0.01, 0.981
MEAS_MODE          = "VEL"          # "VEL" 또는 "ACC"
#SMOOTH_N           = 1
#HPF_CUTOFF         = 5.0
#MAX_FMAX           = 50
REC_DURATION_DEF   = 60.0
FN_BAND            = (5, 50)
BUF_LEN, MIN_LEN   = 8192, 1024      # 실시간 버퍼
USE_SPLIT          = True            # 그래프 3-way 분할
F_MIN = 5

# 4. 사용자 조정 상수
# ────────────────── Hi-Res (고정 정밀) ──────────────────
HIRES = False          # 토글 버튼으로 바뀜
HIRES_LEN_SEC = 8      # 실시간 FFT 길이 (기존 4 → 8 s)
N_TAPER = 4            # 멀티테이퍼 개수
USE_MULTITAPER = True  # False 면 Welch 다중 평균으로 대체

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

def smooth_y(vals, n=None):
    n = n or CFG["SMOOTH_N"]
    if n <= 1 or len(vals) < n:
        return vals[:]
    return np.convolve(vals, np.ones(n)/n, mode="same")

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
        band_rms.append((cen, 20*np.log10(max(rms, REF0*1e-6)/REF0)))
        band_pk.append((cen, 20*np.log10(max(pk,  REF0*1e-6)/REF0)))
    if len(band_rms) >= SMOOTH_N:
        ys = smooth_y([y for _, y in band_rms])
        band_rms = list(zip([x for x, _ in band_rms], ys))
    return band_rms, band_pk


from kivy.uix.modalview import ModalView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput

class ParamPopup(ModalView):
    def __init__(self, app, **kw):
        super().__init__(size_hint=(.9, .7), **kw)
        self.app = app

        # 항목은 “라벨 + (슬라이더+값표시)” 두 칼럼으로 보이도록
        root = GridLayout(cols=2, spacing=12, padding=18)
        self.add_widget(root)

        # ── 정의: (표시이름, CFG-키, 슬라이더 kwargs) ──────────────
        items = [
            ("Band Width (Hz)", "BAND_HZ",     dict(min=0.1, max=2.0,  step=.1)),
            ("HPF Cutoff (Hz)", "HPF_CUTOFF",  dict(min=1.0, max=20.0, step=1)),
            ("Fmax (Hz)",       "MAX_FMAX",    dict(min=20.0, max=200.0, step=5)),
            ("Smooth-N",        "SMOOTH_N",    dict(min=1,   max=5,    step=1)),
        ]

        self.sliders = {}

        for text, key, s_kwargs in items:

            # ① 왼쪽 : 파라미터 이름
            root.add_widget(Label(text=text, size_hint_x=.35, halign='left'))

            # ② 오른쪽 : 슬라이더 + 값 라벨을 한 줄(BoxLayout)에
            line = BoxLayout(orientation='horizontal', spacing=8)

            slider = Slider(**s_kwargs, value=CFG[key], size_hint_x=.75)
            val_lbl = Label(text=f"{CFG[key]:g}",
                            size_hint_x=.25, bold=True)

            # 슬라이더 값이 바뀌면 CFG와 라벨 동시 업데이트
            def _on_val(slider, v, k=key, lbl=val_lbl):
                CFG[k] = type(CFG[k])(v)          # 형 변환 유지
                lbl.text = f"{CFG[k]:g}"

            slider.bind(value=_on_val)

            line.add_widget(slider)
            line.add_widget(val_lbl)
            root.add_widget(line)

            self.sliders[key] = slider

        # ── 하단 APPLY / CLOSE 버튼 ───────────────────────────
        btn = Button(text="APPLY  ⟶  CLOSE", size_hint=(1, .18))
        btn.bind(on_release=self._apply_and_close)
        # GridLayout의 두 칼럼을 꽉 채우도록 colspan 흉내
        root.add_widget(Label())   # 빈 셀
        root.add_widget(btn)

    def _apply_and_close(self, *_):
        # 파라미터가 바뀌었으니 그래프 스케일 갱신
        for g in self.app.graphs:
            g.update_graph(g.datasets, g.diff, g.max_x)
        self.dismiss()

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
    # (기존)  ─ Python 3.10+ 전용 --------------------
    from typing import Optional          # 파일 상단에 한 번만!
   
    def update_graph(self, ds, df, xm, status: Optional[str] = None):
   
    #def update_graph(self, ds, df, xm, status: str | None = None):
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
        self.redraw()


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
            dash_len, gap_len = 1.5, 6.
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
   
        # ① 현재 축 라벨이 전혀 없으면 무조건 다시 만들어야 함
        no_axis_now = not any(getattr(ch, "_axis", False) for ch in self.children)
    
        need_new_axis = (
            (self.max_x, (self.Y_MIN, self.Y_MAX)) != self._prev_ticks
        ) or no_axis_now          # ← 여기 한 줄만 추가!
    
        if need_new_axis:
            # 남아 있던 축 라벨 제거
            for ch in list(self.children):
                if getattr(ch, "_axis", False):
                    self.remove_widget(ch)
            self._make_labels()
            self._prev_ticks = (self.max_x, (self.Y_MIN, self.Y_MAX))
       
        # (피크 라벨·배지 제거 코드는 그대로 두세요)
        for ch in list(self.children):
            if getattr(ch, "_peak", False):
                self.remove_widget(ch)
   
        # ── ② 축 라벨이 바뀌었을 때만 다시 생성 ──
        if need_new_axis:
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


                diff_sorted = sorted(self.diff,
                                     key=lambda p: abs(p[1]), reverse=True)
                for k, (fx, fy) in enumerate(self._select_peaks(diff_sorted)):
                    sx, sy = self._scale([(fx, fy)])[:2]
                    peaks.append((sx, sy, fx, 'diff', k, fy))


            PopMatrix()


        # ── ③ 피크 라벨 배치 -----------------------------------------------
        for pt in peaks:
            if len(pt) == 5:                      # ▸ X / Y / Z 라벨 (기존 형태)
                sx, sy, fx, axis, order = pt
                txt  = f"{fx:.1f} Hz"
                color = self.AXIS_CLR[axis] + (1,)
            else:                                 # ▸ diff 라벨  (Hz + dB)
                sx, sy, fx, axis, order, dv = pt   # axis 값은 'diff' 자리 → 쓰진 않음
                txt  = f"{fx:.1f} Hz\n{dv:+.1f} dB"
                color = (1, 1, 1, 1)               # 흰색
        
            lbl = Label(text=txt, color=color, size_hint=(None, None))
            lbl.texture_update()
            w, h = lbl.texture_size
        
            # 꼭짓점 위에서 조금 띄워, 그래프 영역 안쪽에 배치
            px = self.x + sx - w / 2
            py = self.y + min(sy + 8 + order * 14,
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
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
   
        # ── 0) 상태 라벨 ────────────────────────────────────────────────
        self.label = Label(text="Pick up to 2 CSV / TXT",
                           size_hint=(1, .05), color=(1, 1, 1, 1))
        root.add_widget(self.label)
   
        BTN_H   = dp(48)                 # 버튼 높이(한곳에서 일괄 조정)
        SPACING = 10
   
        # 헬퍼: “두 개씩” 넣는 GridLayout 생성 ---------------------------
        def _row():
            return GridLayout(cols=2, spacing=SPACING,
                              size_hint=(1, None), height=BTN_H)
   
        # ── 1) 파일 & 실행 ------------------------------------------------
        row = _row()
        self.btn_sel = Button(text="Select CSV", height=BTN_H,
                              on_press=self.open_chooser, disabled=True)
        self.btn_run = Button(text="FFT RUN",    height=BTN_H,
                              on_press=self.run_fft,      disabled=True)
        row.add_widget(self.btn_sel);  row.add_widget(self.btn_run)
        root.add_widget(row)
   
        # ── 2) 녹음 & 실시간 --------------------------------------------
        row = _row()
        self.btn_rec = Button(text=f"Record {int(self.REC_DURATION)} s",
                              height=BTN_H, disabled=True,
                              on_press=self.start_recording)
        self.btn_rt  = Button(text="Realtime FFT (OFF)", height=BTN_H,
                              on_press=self.toggle_realtime)
        row.add_widget(self.btn_rec);  row.add_widget(self.btn_rt)
        root.add_widget(row)

        # ── 5) 스피너 두 개도 한 줄 --------------------------------------
        row = _row()
        self.spin_dur = Spinner(text=f"{int(self.REC_DURATION)} s",
                                values=("10 s","30 s","60 s","120 s"),
                                height=BTN_H)
        self.spin_dur.bind(text=lambda s, t:
                           self._set_rec_dur(float(t.split()[0])))
   
        self.spin_sm  = Spinner(text=str(CFG["SMOOTH_N"]),
                                values=("1","2","3","4","5"),
                                height=BTN_H)
        self.spin_sm.bind(text=lambda s, t:
                          self._set_smooth(int(t)))
        row.add_widget(self.spin_dur); row.add_widget(self.spin_sm)
        root.add_widget(row)
   
        # ── 3) Hi-Res & Mode --------------------------------------------
        row = _row()
        self.btn_hires = Button(text="Hi-Res: OFF", height=BTN_H,
                                on_press=self._toggle_hires)
        self.btn_mode  = Button(text=f"Mode: {MEAS_MODE}", height=BTN_H,
                                on_press=self._toggle_mode)
        row.add_widget(self.btn_hires); row.add_widget(self.btn_mode)
        root.add_widget(row)
       
   
        # ── 4) F0 저장 & 파라미터 팝업 -----------------------------------
        row = _row()
        self.btn_setF0 = Button(text="Set F0 (baseline)", height=BTN_H,
                                on_press=self._save_baseline)
        self.btn_param = Button(text="⚙︎ PARAM", height=BTN_H,
                                on_press=lambda *_: ParamPopup(self).open())
        row.add_widget(self.btn_setF0); row.add_widget(self.btn_param)
        root.add_widget(row)
   

   
        # ── 6) 그래프 영역 ----------------------------------------------
        gbox = BoxLayout(orientation='vertical',
                         size_hint=(1, .60), spacing=SPACING)
        self.graphs = []
        for _ in range(3):
            gw = GraphWidget(size_hint=(1, 1/3))
            self.graphs.append(gw)
            gbox.add_widget(gw)
        root.add_widget(gbox)
   
        # ── 7) 퍼미션 체크 예약 ------------------------------------------
        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ───────────────────────────── 헬퍼
    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
   
        # ── 메시지 색상 구분 ──────────────────────
        if msg.startswith("PLZ"):             # 예: "PLZ CHECK"
            self.label.color = (1, 0, 0, 1)   # 빨간색
        elif msg.startswith("GOOD"):
            self.label.color = (0, 1, 0, 1)   # 초록색
        else:
            self.label.color = (1, 1, 1, 1)   # 기본 흰색
   
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


    def _toggle_hires(self, *_):
        global HIRES
        HIRES = not HIRES
        self.btn_hires.text = f"Hi-Res: {'ON' if HIRES else 'OFF'}"
        self.log(f"Hi-Res mode → {'ON' if HIRES else 'OFF'}")

   
    # ──────────────────────────────────────────────────────────────
    #  FFT 실시간 분석 루프 – 최종 수정본
    # ──────────────────────────────────────────────────────────────
    def _rt_fft_loop(self):
        """
        0.5 초마다 가속도 버퍼 스냅샷 → Welch-like 대역 RMS/Peak 계산 → 그래프 갱신
        Hi-Res 모드가 ON이면
          · 분석 길이 = HIRES_LEN_SEC
          · 다중 DPSS 테이퍼(N_TAPER) 평균 사용
        """
        RT_REFRESH_SEC  = 0.5          # UI 갱신 속도
        MIN_FS          = 50           # 최소 유효 샘플링 주파수
        N_KEEP_FS_EST   = 2048         # fs 추정 시 뒤쪽 N개만 사용
   
        try:
            while self.rt_on:
                time.sleep(RT_REFRESH_SEC)
   
                # ① 버퍼 스냅샷 (락 보호) ---------------------------------
                with self._buf_lock:
                    snap = {ax: list(self.rt_buf[ax]) for ax in "xyz"}
   
                axis_sets: dict[str, tuple] = {}
                xmax = 0.0
   
                # ② 축별 FFT 분석 ----------------------------------------
                for ax in "xyz":
                    if len(snap[ax]) < MIN_FS * (HIRES_LEN_SEC if HIRES else 4):
                        continue
   
                    # ── robust fs 추정
                    t_arr, val_arr, _ = zip(*snap[ax])
                    fs = robust_fs(t_arr[-N_KEEP_FS_EST:])
                    if fs is None or fs < MIN_FS:
                        continue
   
                    # ── FFT 길이 : 2^n 로 패딩
                    seg_len_sec = HIRES_LEN_SEC if HIRES else 4
                    fft_len     = next_pow2(int(fs * seg_len_sec))
                    if len(val_arr) < fft_len:
                        continue      # 샘플 부족
   
                    # 최근 fft_len 구간 추출
                    sig = np.asarray(val_arr[-fft_len:], float)
                    sig = sig - sig.mean()        # DC 제거
   
                    # ── 창 & 스펙트럼 ---------------------------------
                    if HIRES and USE_MULTITAPER and ss is not None:
                        tapers  = ss.windows.dpss(fft_len, NW=2.5,
                                                  Kmax=N_TAPER, sym=False)
                        spec_sq = 0.0
                        for tap in tapers:
                            spec_sq += np.abs(np.fft.rfft(sig * tap))**2
                        amp_a = (spec_sq / N_TAPER)**0.5 * 2 / (fft_len*np.sqrt(2))
                    else:
                        win   = np.hanning(fft_len)
                        amp_a = 2 * np.abs(np.fft.rfft(sig * win)) / (fft_len*np.sqrt(2))
   
                    freq    = np.fft.rfftfreq(fft_len, d=1/fs)
                    amp_lin, ref0 = acc_to_spec(freq, amp_a)
   
                    # ── 0.5 Hz 밴드 RMS / Peak -------------------------
                    rms_line, pk_line = [], []
                    FMAX = min(MAX_FMAX, fs * 0.5)
                   # ① Welch 루프 대역폭
                    for lo in np.arange(CFG["HPF_CUTOFF"], FMAX, CFG["BAND_HZ"]):

                        hi  = lo + BAND_HZ
                        sel = (freq >= lo) & (freq < hi)
                        if not sel.any():
                            continue
                        cen = (lo + hi) * 0.5
                        rms = np.sqrt(np.mean(amp_lin[sel]**2))
                        pk  = amp_lin[sel].max()
                        rms_line.append((cen, 20*np.log10(max(rms, ref0*1e-6)/ref0)))
                        pk_line .append((cen, 20*np.log10(max(pk , ref0*1e-6)/ref0)))
   
                    if not rms_line:
                        continue
   
                    # ── 스무딩
                    if len(rms_line) >= CFG["SMOOTH_N"]:
                        ys = smooth_y([y for _, y in rms_line])
                        rms_line = [(x, y) for (x, _), y in zip(rms_line, ys)]
                       
                    # ── 공진 주파수(Fₙ) 추적 ---------------------------
                    f_centres = np.array([x for x, _ in rms_line])
                    f_vals    = np.array([y for _, y in rms_line])
                    band_sel  = (f_centres >= FN_BAND[0]) & (f_centres <= FN_BAND[1])
                    if band_sel.any():
                        self.last_fn = f_centres[band_sel][f_vals[band_sel].argmax()]
   
                    # ── 결과 저장
                    axis_sets[ax] = (rms_line, pk_line, ax)
                    xmax = max(xmax, FMAX)
   
                # ③ UI 스레드로 그래프 업데이트 ---------------------------
                if axis_sets:
                    def _update(_dt, sets=axis_sets, xm=xmax):
                        for ax, g in zip("xyz", self.graphs):
                            if ax in sets:
                                g.update_graph([sets[ax]], [], xm, "")   # status 없음
                            else:
                                g.update_graph([], [], xm, "")           # 빈 그래프
                    Clock.schedule_once(_update)
   
        except Exception:
            Logger.exception("Realtime FFT thread crashed")
            self.rt_on = False
            Clock.schedule_once(
                lambda *_: setattr(self.btn_rt, "text", "Realtime FFT (OFF)"))
    # ───────────────────────────── CSV FFT (백그라운드)
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()



    # ────────────────────────────────────────────
    #  CSV 2-개 FFT  → ①RMS/Peak 2 세트 ②RMS 차이(dB)
    # ────────────────────────────────────────────
    def _fft_bg(self):
   
        # ===== 1) 설정 =========================================================
        HIRES_CSV       = True          # ← 고정밀 모드 ON/OFF
        PAD_FACTOR_HI   = 4             # ← 4 × 패딩 (Δf ≈ 1/4 로 줄어듦)
        BAND_W          = BAND_HZ       # 대역폭(0.5 Hz) – 전역 상수 그대로 사용
        # ======================================================================
   
        try:
            if len(self.paths) < 2:
                raise ValueError("SELECT 2 CSV FILE")
   
            data, xmax = [], 0.0
   
            # ── 각 CSV 파일 개별 처리 ───────────────────────────────────────
            for path in self.paths[:2]:
                t, a = self._load_csv(path)
                if t is None:
                    raise ValueError(f"{os.path.basename(path)} : CSV PARSE FAIL")
   
                # ── ① 샘플링 주파수 추정 및 최대 분석 주파수 설정 ───────────
                fs = robust_fs(t)                # ← 타임스탬프 배열 그대로!
                if fs is None:
                    raise ValueError("UNSTABLE SAMPLING RATE")
   
                nyq  = fs * 0.5
                FMAX = max(HPF_CUTOFF + BAND_W, min(nyq, MAX_FMAX))
   
                # ── ② FFT 신호 준비 (윈도우 + 제로패딩) ────────────────────
                pad_len = next_pow2(len(a)) * (PAD_FACTOR_HI if HIRES_CSV else 1)
                win     = np.hanning(pad_len)
                sig_pad = np.zeros(pad_len, float)
                sig_pad[:len(a)] = a - a.mean()         # DC 제거
   
                spec  = np.fft.fft(sig_pad * win)
                amp_a = 2 * np.abs(spec[:pad_len//2]) / (pad_len*np.sqrt(2))
                freq  = np.fft.fftfreq(pad_len, d=1/fs)[:pad_len//2]
   
                # ── ③ 사용 대역 잘라내기 ───────────────────────────────────
                sel = (freq >= HPF_CUTOFF) & (freq <= FMAX)
                freq, amp_a = freq[sel], amp_a[sel]
                amp_lin, REF0 = acc_to_spec(freq, amp_a)
   
                # ── ④ 0.5 Hz 밴드 RMS / PEAK(dB) 계산 ────────────────────
                rms_line, pk_line = [], []
                for lo in np.arange(HPF_CUTOFF, FMAX, BAND_W):
                    hi  = lo + BAND_W
                    m   = (freq >= lo) & (freq < hi)
                    if not m.any():
                        continue
                    cen = (lo + hi) * 0.5
                    rms = np.sqrt(np.mean(amp_lin[m]**2))
                    pk  = amp_lin[m].max()
                    rms_line.append((cen, 20*np.log10(max(rms, REF0*1e-6)/REF0)))
                    pk_line .append((cen, 20*np.log10(max(pk , REF0*1e-6)/REF0)))
   
                # ── ⑤ 스무딩 ──────────────────────────────────────────────
                # ── 스무딩 블록 (두 곳 모두) ────────────────
                if len(rms_line) >= CFG["SMOOTH_N"]:
                    ys = smooth_y([y for _, y in rms_line])
                    rms_line = [(x, y) for (x, _), y in zip(rms_line, ys)]
                   
                data.append((rms_line, pk_line))
                xmax = max(xmax, FMAX)
   
            # ===== 2) 두 파일 간 RMS 차이(dB) 계산 =========================
            f1 = dict(data[0][0]);  f2 = dict(data[1][0])
            diff_line = [(f, f1[f] - f2[f]) for f in sorted(set(f1) & set(f2))]

            # |dB| 가 큰 순서 Top-3 추출
            top3 = sorted(diff_line, key=lambda p: abs(p[1]), reverse=True)[:3]
            peak_txt = "  •  ".join(f"{f:4.1f} Hz : {d:+.1f} dB" for f, d in top3)
   
            # ===== 3) 15 dB 이상이면 PLZ CHECK 배지 띄우기 ================
            alert_msg = ("PLZ CHECK" if any(abs(d) >= 10 for _, d in diff_line)
                         else "GOOD")
   
            # ===== 4) UI 스레드에 결과 갱신 =================================
            def _update(_dt):
                rms0, pk0 = data[0]
                rms1, pk1 = data[1]
   
                self.graphs[0].update_graph([(rms0, pk0, 'x')], [], xmax)
                self.graphs[1].update_graph([(rms1, pk1, 'y')], [], xmax)
                self.graphs[2].update_graph([], diff_line, xmax, alert_msg)
   
                # ★ 화면 하단 메시지 = 경고(or GOOD) + Top-3 정보
                self.log(f"{alert_msg}   |   TOP Δ: {peak_txt}")
   
            Clock.schedule_once(_update)
   
        except Exception as exc:
            self.log(f"FFT ERROR : {exc}")
   
        finally:
            Clock.schedule_once(lambda *_:
                setattr(self.btn_run, "disabled", False))
               

    # ───────────────────────────── CSV 로드
    def _load_csv(self, path):
        num_re = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?$")
        try:
            t,a = [],[]
            with open(path, encoding="utf-8", errors="replace") as f:
                sample=f.read(2048); f.seek(0)
                try:
                    dialect=csv.Sniffer().sniff(sample, delimiters=";, \t")
                except csv.Error:
                    dialect=csv.excel
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
                                              multiple=True, mime_type="*/*")
                    return
                except Exception:
                    pass
            # plyer fallback
            filechooser.open_file(on_selection=self.on_choose,
                                  multiple=True, filters=["*.csv", "*.txt", "*.*"],
                                  path=DOWNLOAD_DIR)
            return
   
        # ---------- 2) Windows / macOS / Linux ----------
        try:
            # Tkinter는 파이썬 표준 라이브러리라 별도 설치 불필요
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()        # 창 감춤
            paths = filedialog.askopenfilenames(
                title="Select up to 2 data files",
                filetypes=[("CSV files", "*.csv *.txt"), ("All files", "*,*")])
            # Tk 객체 해제
            root.update(); root.destroy()
            self.on_choose(list(paths)[:3])
        except Exception as e:
            # 그래도 실패하면 Kivy FileChooserPopup으로 최종 fallback
            Logger.warning(f"Tk file dialog error: {e}")
            filechooser.open_file(on_selection=self.on_choose,
                                  multiple=True, filters=["*.csv", "*.txt"])

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
 
