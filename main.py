"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판
+ 30 초 실시간 가속도 기록 (Downloads 폴더 저장 개선판)
"""
# ── 표준 & 3rd-party ────────────────────────────────────────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time, re
import numpy as np
from collections import deque
from numpy.fft import fft

from plyer import accelerometer                # 센서

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
from plyer               import filechooser     # (SAF 실패 시 fallback)
from kivy.uix.spinner import Spinner

# ---------- 사용자 조정값 ---------- #
BAND_HZ     = 2.0
REF_MM_S    = 0.01
REF_ACC = 0.981
MEAS_MODE = "VEL"
PEAK_COLOR  = (1,1,1)
SMOOTH_N = 2
HPF_CUTOFF = 5.0
REC_DURATION_DEFAULT = 60.0

# 공진 탐색 범위 ↓ (기존 (5,25) → 상한 50 Hz 로 확대)
FN_BAND     = (5, 50)   # ← 이렇게만 변경
THR_DF      = 0.5       # ΔF 경고 임계값 (필요 시 그대로)
# ----------------------------------- #

BUF_LEN   = 2048       # Realtime 버퍼 길이
MIN_LEN   = 1024          # FFT 돌리기 전 최소 샘플 수


# ── Android 전용 모듈 ───────────────────────────────────────────────
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
        check_permission = lambda *a, **kw: True
        request_permissions = lambda *a, **kw: None
        class _P:     # 빌드오저 recipe 미포함 시 더미
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
        Permission = _P
    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
        # ★ 공식 Downloads 절대경로 가져오기
        Environment = autoclass("android.os.Environment")
        DOWNLOAD_DIR = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
    except Exception:
        ANDROID_API = 0
        DOWNLOAD_DIR = "/sdcard/Download"
else:                                   # 데스크톱 테스트용
    DOWNLOAD_DIR = os.path.expanduser("~/Download")

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


# ── SAF URI → 로컬 파일 경로 ────────────────────────────────────────
def uri_to_file(u: str) -> str | None:
    """
    content://, file://, 절대경로 모두 → 실제 파일로 복사(또는 검증) 후 경로 반환
    실패 시 None
    """
    if not u:
        return None

    # ① file:// 스킴
    if u.startswith("file://"):
        real = urllib.parse.unquote(u[7:])      # file:///sdcard/... → /sdcard/...
        return real if os.path.exists(real) else None

    # ② content:// (SAF)
    if u.startswith("content://") and ANDROID and SharedStorage:
        try:
            # Downloads/fft_tmp_xxx.csv 로 임시 복사
            dst = SharedStorage().copy_from_shared(
                    u, uuid.uuid4().hex+".csv", to_downloads=True)
            return dst
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
            return None

    # ③ 이미 절대경로인 경우
    if os.path.exists(u):
        return u

    # ④ 마지막 Fallback ― 권한 부족 등으로 열 수 없음
    Logger.warning(f"uri_to_file: cannot access {u}")
    return "NO_PERMISSION"        # 호출자에게 권한 문제임을 알림

def dashed_line(canvas, pts, dash=8, gap=6, **kw):
    """
    pts=[x1,y1,x2,y2,…] 를 (dash, gap) 패턴으로 잘라 그린다.
    OpenGL-ES(안드로이드)에서도 동작하는 ‘가짜 점선’ 구현.
    """
    if len(pts) < 4:
        return
    for i in range(0, len(pts)-2, 2):
        x1, y1, x2, y2 = pts[i:i+4]
        seg_len = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        if seg_len == 0:
            continue
        nx, ny = (x2-x1)/seg_len, (y2-y1)/seg_len
        s, draw = 0.0, True
        while s < seg_len:
            length = min(dash if draw else gap, seg_len - s)
            if draw:
                Line(points=[x1+nx*s, y1+ny*s,
                             x1+nx*(s+length), y1+ny*(s+length)],
                     **kw)
            s += length
            draw = not draw

# ★★★ ① 추가 : 공용 변환 함수 ★★★
def acc_to_spec(freq, amp_a):
    """
    가속도 스펙트럼 → ( MEAS_MODE 에 따라 )
      • VEL :  속도 [mm/s RMS]   (0 dB 기준 = REF_MM_S)
      • ACC :  가속도 [m/s² RMS] (0 dB 기준 = 0.981 ≒ 0.1 g)
    return  (amp, ref)  : 선형 스펙트럼값, 0 dB 기준값
    """
    if MEAS_MODE == "VEL":            # 그대로
        f_nz = np.where(freq < 1e-6, 1e-6, freq)
        amp  = amp_a / (2*np.pi*f_nz) * 1e3
        ref  = REF_MM_S
    else:                             # "ACC"
        amp  = amp_a
        ref  = REF_ACC               # ← 여기!
    return amp, ref
# ★★★ ① 끝 ★★★



# ── 공통 스무딩 함수 ─────────────────
def smooth_y(vals, n=None):
    """n-point moving-average; n==1 ➜ no smoothing"""
    if n is None:            # ← 호출자가 n을 안 줘도 되도록
        n = SMOOTH_N
    if n <= 1 or len(vals) < n:
        return vals[:]
    kernel = np.ones(n)/n
    return np.convolve(vals, kernel, mode="same")


# ── 그래프 위젯 (Y축 고정 · 세미로그 · 좌표 캐스팅) ───────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30

    #            0        1        2        3        4        5
    COLORS   = [(1,0,0), (1,1,0), (0,0,1), (0,1,1), (0,1,0), (1,0,1)]
    #            빨강     노랑     파랑     시안     초록     자홍
    DIFF_CLR = (1,1,1)          # 두 CSV 차이선은 흰색
    LINE_W   = 2.5

    Y_TICKS = [0, 40, 80, 150]
    Y_MAX   = Y_TICKS[-1]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = 1
        self.bind(size=self.redraw)

    # ---------- 외부 호출 ----------

    def update_graph(self, ds, df, xm, ym_est):
        self.max_x   = max(1e-6, float(xm))
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []

        # --- 새로: 그래프 전체 y-범위 스캔 -----------------
        ys = []
        for seq in self.datasets + [self.diff]:
            ys.extend(y for _, y in seq)

        if ys:            # 데이터가 있을 때만
            max_y = max(ys)
            min_y = min(ys)
        else:             # 비어 있으면 기본값
            max_y, min_y = 0, 0

        # tick 간격 20 dB 로 라운드
        top  = max(20, ((int(max_y) // 20) + 1) * 20)
        low  = ((int(min_y) // 20) - 1) * 20      # 음수 tick 포함

        self.Y_TICKS = list(range(low, top + 1, 20))
        self.Y_MIN   = low
        self.Y_MAX   = top
        self.redraw()


    def y_pos(self, v: float) -> float:
        """
        입력 dB 값 → 화면 y 좌표
        (Y_MIN ~ Y_MAX 범위를 PAD_Y ~ height-PAD_Y 로 선형 매핑)
        """
        h = self.height - 2*self.PAD_Y          # ← 높이 먼저 계산
        if h <= 0:
            return self.PAD_Y                   # 안전장치

        # 클램핑
        v = max(self.Y_MIN, min(v, self.Y_MAX))

        # 선형 변환(하단 PAD_Y → 상단 height-PAD_Y)
        return self.PAD_Y + (v - self.Y_MIN) / (self.Y_MAX - self.Y_MIN) * h
    
    # ---------- 좌표 변환 ----------
    # ---------- 좌표 변환 ----------
    def _scale(self, pts):
        """
        (주파수[Hz], dB) 목록 → [x1, y1, x2, y2, …]  (캔버스 좌표계)
        self._f(), h 변수 등을 사용하지 않고
        GraphWidget.y_pos() 만 이용해 변환한다.
        """
        w = float(self.width) - 2 * self.PAD_X
        out = []
        for x, y in pts:
            sx = self.PAD_X + (float(x) / self.max_x) * w      # X축 선형
            sy = self.y_pos(float(y))                          # Y축 선형(전체-범위)
            out += [sx, sy]
        return out   

    # ---------- 그리드 ----------
    def _grid(self):
        """세로 그리드: 0 ~ 50 Hz / 10 Hz 간격"""
        n_tick = int(self.max_x // 10) + 1
        span   = max(n_tick - 1, 1)
        gx     = (self.width - 2*self.PAD_X) / span

        Color(.6, .6, .6)
        for i in range(n_tick):
            Line(points=[self.PAD_X + i*gx, self.PAD_Y,
                         self.PAD_X + i*gx, self.height - self.PAD_Y])

        # 가로선은 그대로
        for v in self.Y_TICKS:
            y = self._scale([(0, v)])[1]
            Line(points=[self.PAD_X, y,
                         self.width - self.PAD_X, y])

    # ---------- 축 라벨 ----------
    def _labels(self):
        """세로 눈금 라벨을 그린다."""
        # ── 예전 축 라벨 지우기 ──
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # ── X-축 (10 Hz 간격) ──────────────────
        n_tick = int(self.max_x // 10) + 1          # 최소 1
        span   = max(n_tick - 1, 1)                 # 0 나눗셈 방지
        for i in range(n_tick):
            x = self.PAD_X + i * (self.width - 2*self.PAD_X) / span - 20
            lbl = Label(text=f"{10*i} Hz",
                        size_hint=(None, None), size=(60, 20),
                        pos=(x, self.PAD_Y - 28))
            lbl._axis = True
            self.add_widget(lbl)

        # ── Y-축 (20 dB 간격) ──────────────────
        for v in self.Y_TICKS:
            y = self._scale([(0, v)])[1] - 8
            for x_pos in (self.PAD_X - 68, self.width - self.PAD_X + 10):
                lbl = Label(text=f"{v}",
                            size_hint=(None, None), size=(60, 20),
                            pos=(x_pos, y))
                lbl._axis = True
                self.add_widget(lbl)

    
    # ── 새로 추가 :  이전 축·피크 라벨 싹 지우기 ──────────────────
    def _clear_labels(self):
        """
        self.children 중  _axis / _peak  플래그가 달린 위젯을
        전부 제거한다.  ( redraw() 가 부를 때마다 호출 )
        """
        for w in list(self.children):
            if getattr(w, "_axis", False) or getattr(w, "_peak", False):
                self.remove_widget(w)

    
    # ---------- 메인 그리기 ---------
    def redraw(self, *_):
        # ① 캔버스·라벨 초기화
        self.canvas.clear()
        self._clear_labels()     # ← 새로 만든 함수로 한 번에 정리

        if not self.datasets:    # 데이터 없으면 끝
            return

        # ② 그리드 + 라벨 다시 그리기
        with self.canvas:
            self._grid()
        self._labels()           # 라벨은 Widget 으로 children 에 추가

        # ③ 데이터 라인 & 피크
        peaks = []
        with self.canvas:
            for idx, pts in enumerate(self.datasets):
                if not pts:
                    continue
                axis_idx = idx // 2
                Color(*self.COLORS[axis_idx % len(self.COLORS)])
                scaled = self._scale(pts)

                # Peak(점선) / RMS(실선) 구분
                if idx % 2:
                    dashed_line(self.canvas, scaled, dash=10, gap=6, width=self.LINE_W)
                else:
                    Line(points=scaled, width=self.LINE_W)
                    try:
                        fx, fy = max(pts, key=lambda p: p[1])
                        sx, sy = self._scale([(fx, fy)])[0:2]
                        peaks.append((fx, fy, sx, sy))
                    except ValueError:
                        pass

            # diff 라인이 있으면 추가
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # ④ 피크 주파수 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None, None), size=(85, 22),
                        pos=(sx-28, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # ⑤ ΔF 표시
        if len(peaks) >= 2:
            delta = abs(peaks[0][0] - peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            info  = Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                          size_hint=(None,None), size=(190,24),
                          pos=(self.PAD_X, self.height-self.PAD_Y+6),
                          color=clr)
            info._peak = True
            self.add_widget(info)

        # ⑥ 실시간 Fₙ 표시
        app = App.get_running_app()
        if getattr(app, "last_fn", None) is not None:
            lbl = Label(text=f"Fₙ={app.last_fn:.2f} Hz",
                        size_hint=(None,None), size=(115,22),
                        pos=(self.width-155, self.height-28),
                        color=(1,1,0,1))
            lbl._peak = True
            self.add_widget(lbl)


# ── 메인 앱 ────────────────────────────────────────────────────────
class FFTApp(App):

    OFFSET_DB = 20 
    def __init__(self, **kw):
        super().__init__(**kw)
        # 실시간 FFT
        self.rt_on = False

        self.rt_buf = {ax: deque(maxlen=BUF_LEN) for ax in ('x','y','z')}


        # 60 초 기록
        self.rec_on = False
        self.rec_start = 0.0
        self.rec_files = {}
        self.REC_DURATION = REC_DURATION_DEFAULT   # 필요 시 메뉴로 수정
        
        self.F0 = None      # ⊕ 기준 공진수
        self.last_fn = None #   실시간 Fₙ 임시보

    # ---------------  FFTApp 클래스 안  ----------------
    def _set_rec_dur(self, spinner, txt):
        """Spinner 콜백 – 녹음 길이 변경"""
        self.REC_DURATION       = float(txt.split()[0])
        self.btn_rec.text       = f"Record {int(self.REC_DURATION)} s"
        self.log(f"▶ 녹음 길이 {self.REC_DURATION:.0f} s 로 설정")
    # ---------------------------------------------------
    # ─────────  FFTApp 내부  ─────────
    def _set_smooth(self, spinner, txt):
        """Spinner 콜백 – 스무딩 창 크기 변경"""
        global SMOOTH_N
        SMOOTH_N        = int(txt)          # '1' → 1,  '3' → 3 …
        self.log(f"▶ 스무딩 창 = {SMOOTH_N} point")
    # ────────────────────────────────

    # ── 공통 로그 ───────────────────────────────────────────────
    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ── 권한 체크 ───────────────────────────────────────────────
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            self.btn_rec.disabled = False
            return
        need=[Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        MANAGE = getattr(Permission,"MANAGE_EXTERNAL_STORAGE",None)
        if MANAGE: need.append(MANAGE)
        if ANDROID_API>=33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]
        def _cb(perms,grants):
            ok = any(grants)
            self.btn_sel.disabled = not ok
            self.btn_rec.disabled = not ok
            if not ok:
                self.log("저장소 권한 거부 – 파일 접근/저장이 제한됩니다")
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
            self.btn_rec.disabled = False
        else:
            request_permissions(need,_cb)

        #   ★ 여기 넣어 두면 다른 메서드들이 쉽게 호출
    # ──────────────────────────────────────────────
    def _has_allfiles_perm(self) -> bool:
        """
        Android 11+ ‘모든-파일’(MANAGE_EXTERNAL_STORAGE) 권한 보유 여부
        * Android 10 이하는 해당 권한 자체가 없으므로 True 반환
        """
        MANAGE = getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)
        return not MANAGE or check_permission(MANAGE)

    # ──────────────────────────────────────────────────────────
    # ① 30 초 가속도 기록 기능
    # ──────────────────────────────────────────────────────────
    def start_recording(self,*_):
        if self.rec_on:
            self.log("이미 기록 중입니다"); return
        try:
            accelerometer.enable()
        except (NotImplementedError,Exception) as e:
            self.log(f"센서 사용 불가: {e}"); return
        # ★ 저장 폴더: 기기별 실제 Downloads 경로
        save_dir = DOWNLOAD_DIR
        ok=True
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            os.makedirs(save_dir, exist_ok=True)
            self.rec_files={}
            for ax in ('x','y','z'):
                path=os.path.join(save_dir, f"acc_{ax}_{ts}.csv")
                f=open(path,"w",newline="",encoding="utf-8")
                csv.writer(f).writerow(["time","acc"])
                self.rec_files[ax]=f
            self.log(f"📥 저장 위치: {save_dir}")
        except Exception as e:
            self.log(f"파일 열기 실패: {e}")
            ok=False
        if not ok:
            try: accelerometer.disable()
            except Exception: pass
            return
        self.rec_on=True
        self.rec_start=time.time()
        self.btn_rec.disabled=True
        self.label.text = f"Recording 0/{int(self.REC_DURATION)} s …"
        Clock.schedule_interval(self._record_poll, 1 / 50.0)  # 센서 읽기 간격: 50Hz 기준
        Clock.schedule_once(self._stop_recording, self.REC_DURATION)

    def _record_poll(self, dt):
        if not self.rec_on:
            return False
    
        try:
            ax, ay, az = accelerometer.acceleration
        except Exception as e:
            Logger.warning(f"acc read fail: {e}")
            ax = ay = az = None
    
        if None in (ax, ay, az):
            self.label.text = "Waiting for sensor…"
            return True
    
        now = time.time()
    
        # 유효한 센서 값이 처음 들어온 시점에서부터 rec_start를 설정
        if self.rec_start is None:
            self.rec_start = now
            elapsed = 0
        else:
            elapsed = now - self.rec_start
    
        # CSV 기록
        for ax_name, val in (('x', ax), ('y', ay), ('z', az)):
            csv.writer(self.rec_files[ax_name]).writerow([elapsed, val])
    
        # 레코딩 진행 상태 표시 (0.5초마다 업데이트)
        if int(elapsed * 2) % 2 == 0:
            self.label.text = f"Recording {elapsed:4.1f}/{int(self.REC_DURATION)} s …"
    
        if elapsed >= self.REC_DURATION:
            self._stop_recording()
            return False
    
        return True

    # ────────────── (선택) _stop_recording 정리 ──────────────
    def _stop_recording(self, *_):
        if not self.rec_on:
            return
        for f in self.rec_files.values():
            try:
                f.close()
            except Exception:
                pass
        self.rec_files.clear()
        self.rec_on = False
        self.btn_rec.disabled = False
        self.rec_start = None  # 다음 레코딩을 위해 초기화
    
        if not self.rt_on:
            try:
                accelerometer.disable()
            except Exception:
                pass
    
        self.label.text = "✅ Recording complete!"
        self.log(f"✅ {int(self.REC_DURATION)} s recording finished!")

    # ──────────────────────────────────────────────────────────
    # ② 실시간 FFT (기존)
    # ──────────────────────────────────────────────────────────
    def toggle_realtime(self,*_):
        self.rt_on = not self.rt_on
        self.btn_rt.text=f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try: accelerometer.enable()
            except (NotImplementedError,Exception) as e:
                self.log(f"센서 사용 불가: {e}")
                self.rt_on=False
                self.btn_rt.text="Realtime FFT (OFF)"; return
            Clock.schedule_interval(self._poll_accel, 0)
            threading.Thread(target=self._rt_fft_loop, daemon=True).start()
        else:
            try: accelerometer.disable()
            except Exception: pass

        # ────────────── FFTApp._poll_accel (발췌) ──────────────
        def _poll_accel(self, dt):
            if not self.rt_on:
                return False
            try:
                ax, ay, az = accelerometer.acceleration
                if None in (ax, ay, az):
                    return
    
                now = time.time()
                prev = self.rt_buf['x'][-1][0] if self.rt_buf['x'] else now - dt
                dt_samp = now - prev
    
                # ✅ rec_start 기준 상대시간 사용
                rel_time = now - self.rec_start if self.rec_start else 0
    
                def push(axis, raw):
                    self.rt_buf[axis].append((now, raw, dt_samp))
                    if self.rec_on and axis in self.rec_files:
                        # ✅ 상대 시간으로 기록
                        csv.writer(self.rec_files[axis]).writerow([rel_time, raw])
    
                push('x', abs(ax))
                push('y', abs(ay))
                push('z', abs(az))
    
            except Exception as e:
                Logger.warning(f"acc read fail: {e}")
    # ─────────────────────────────────────────────────────
    #  실시간 FFT 루프 – 2 Hz 대역별 ①RMS + ②피크(dB) 표시
    # ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────
    #  실시간 FFT 루프 – 2 Hz 대역 RMS·Peak 표시
    # ─────────────────────────────────────────────────────
    def _rt_fft_loop(self):
        try:
            while self.rt_on:
                time.sleep(0.5)
    
                if any(len(self.rt_buf[ax]) < MIN_LEN for ax in ('x','y','z')):
                    continue
    
                datasets, ymax, xmax = [], 0, 50
                for axis in ('x','y','z'):
                    ts, val, dt_arr = zip(*self.rt_buf[axis])   # Δt 포함해서 꺼내기
                    #dt = np.mean(dt_arr)                        # 실제 평균 샘플 주기
                    dt = np.median(dt_arr[-128:])
                    if dt <= 0:
                        continue
    
                    sig = (np.asarray(val, float) - np.mean(val)) * np.hanning(len(val))

    
                    # --- FFT 이후 코드는 동일 -------------------
                    raw  = np.fft.fft(sig)
                    amp_a= 2*np.abs(raw[:len(val)//2])/(len(val)*np.sqrt(2))
                    freq = np.fft.fftfreq(len(val), d=dt)[:len(val)//2]
    
                    # 5 Hz HPF + 50 Hz LPF
                    msel = (freq >= HPF_CUTOFF) & (freq <= 50)
                    freq, amp_a = freq[msel], amp_a[msel]
                    if freq.size == 0:
                        continue
    
                    amp_lin, REF0 = acc_to_spec(freq, amp_a)
                    band_rms, band_pk = [], []
                    for lo in np.arange(HPF_CUTOFF, 50, BAND_HZ):
                        hi  = lo + BAND_HZ
                        sel = (freq >= lo) & (freq < hi)
                        if not sel.any():
                            continue
                        rms = np.sqrt(np.mean(amp_lin[sel]**2))
                        pk  = amp_lin[sel].max()
                        centre = (lo+hi)/2
                        band_rms.append((centre, 20*np.log10(max(rms, REF0*1e-4)/REF0)))
                        band_pk .append((centre, 20*np.log10(max(pk , REF0*1e-4)/REF0)))
    
                    # _rt_fft_loop  안
                    if len(band_rms) >= SMOOTH_N:
                        y_sm = smooth_y([y for _, y in band_rms])   # ← 두 번째 인자 생략 OK
                        band_rms = list(zip([x for x, _ in band_rms], y_sm))
    
                    # 공진수 추적
                    loF, hiF = FN_BAND
                    if band_rms:
                        c = np.array([x for x,_ in band_rms])
                        m = np.array([y for _,y in band_rms])
                        selF = (c >= loF) & (c <= hiF)
                        if selF.any():
                            self.last_fn = c[selF][m[selF].argmax()]
    
                    datasets += [band_rms, band_pk]
                    ymax = max(ymax,
                               max(y for _,y in band_rms),
                               max(y for _,y in band_pk))
    
                # ΔF 경고·그래프 갱신 부분 동일 …
                Clock.schedule_once(
                    lambda *_: self.graph.update_graph(datasets, [], xmax, ymax))
    
        except Exception:
            Logger.exception("Realtime FFT thread crashed")
            self.rt_on = False
            Clock.schedule_once(lambda *_: setattr(self.btn_rt, 'text',
                                                   'Realtime FFT (OFF)'))


    
    # ── UI 구성 ───────────────────────────────────────────────
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
    
        # ── 상단 안내 라벨 ────────────────────────────────
        self.label = Label(text="Pick 1 or 2 CSV files", size_hint=(1, .05))
        root.add_widget(self.label)
    
        # ── 파일/실행/녹음 버튼 ───────────────────────────
        self.btn_sel = Button(text="Select CSV", disabled=True,
                              size_hint=(1, .05), on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN", disabled=True,
                              size_hint=(1, .05), on_press=self.run_fft)
        self.btn_rec = Button(text=f"Record {int(self.REC_DURATION)} s",
                              disabled=True, size_hint=(1, .05),
                              on_press=self.start_recording)
    
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(self.btn_rec)
    
        # ── 녹음 길이 Spinner ─────────────────────────────
        self.spin_dur = Spinner(text=f"{int(self.REC_DURATION)} s",
                                values=('10 s', '30 s', '60 s', '120 s'),
                                size_hint=(1, .05))
        self.spin_dur.bind(text=self._set_rec_dur)
        root.add_widget(self.spin_dur)
    
        # ── 측정 모드 토글 ────────────────────────────────
        self.btn_mode = Button(text=f"Mode: {MEAS_MODE}", size_hint=(1, .05),
                               on_press=self._toggle_mode)
        root.add_widget(self.btn_mode)
    
        # ── 기준 F₀ / Realtime 토글 ───────────────────────
        self.btn_setF0 = Button(text="Set F₀ (baseline)",
                                size_hint=(1, .05), on_press=self._save_baseline)
        self.btn_rt = Button(text="Realtime FFT (OFF)", size_hint=(1, .05),
                             on_press=self.toggle_realtime)
    
        # build() 안 — 레이아웃 구성 중
        # (1) 스무딩 Spinner 생성
        self.spin_sm = Spinner(
                text=str(SMOOTH_N),
                values=('1','2','3','4', '5'),     # 필요 수치만 넣으세요
                size_hint=(1, .05))
        self.spin_sm.bind(text=self._set_smooth)
        
        # (2) 원하는 위치에 add_widget
        root.add_widget(self.spin_sm)
        
        root.add_widget(self.btn_setF0)
        root.add_widget(self.btn_rt)
    
        # ── 그래프 ────────────────────────────────────────
        self.graph = GraphWidget(size_hint=(1, .50))
        root.add_widget(self.graph)
    
        # ── 권한 확인 트리거 ──────────────────────────────
        Clock.schedule_once(self._ask_perm, 0)
        return root

    def _toggle_mode(self, *_):
        global MEAS_MODE
        MEAS_MODE = "ACC" if MEAS_MODE == "VEL" else "VEL"
        self.btn_mode.text = f"Mode: {MEAS_MODE}"
        self.log(f"▶ Change the measure mode → {MEAS_MODE}")


    # ⊕ 버튼 콜백
    def _save_baseline(self,*_):
        if self.last_fn is None:
            self.log("X don't know Fₙ ")
        else:
            self.F0 = self.last_fn
            self.log(f"Main Resonance Freq F₀ = {self.F0:.2f} Hz SAVE")



    
    # ── CSV 선택 & FFT 실행 (기존) ───────────────────────────────
    def open_chooser(self, *_):
        """
        CSV 선택 다이얼로그.
        1) Android 11+ 이면서 MANAGE_EXTERNAL_STORAGE 권한이 없는 경우
           → 안내 팝업으로 ‘모든-파일’ 권한 설정 창 유도
        2) 권한이 이미 있거나 데스크톱/구버전 OS
           → Downloads 경로에 직접 접근 (filechooser)
           실패 -or- 권한 거부 → SAF picker 로 우회
        """
    
        # ── Android 11+ : ‘모든-파일’ 권한 확인 ────────────────────
        if ANDROID and ANDROID_API >= 30 and not self._has_allfiles_perm():
            try:
                from jnius import autoclass
                Env = autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():   # 실제로 거부 상태?
                    mv  = ModalView(size_hint=(.8, .35))
                    box = BoxLayout(orientation='vertical', spacing=10, padding=10)
                    box.add_widget(Label(
                        text="⚠️ CSV 파일에 접근하려면\n'모든 파일' 권한이 필요합니다.",
                        halign="center"))
                    box.add_widget(Button(text="권한 설정으로 이동",
                        size_hint=(1, .4),
                        on_press=lambda *_: (mv.dismiss(),
                                             self._goto_allfiles_permission())))
                    mv.add_widget(box)
                    mv.open()
                    return
            except Exception:         # jnius 오류 등
                Logger.exception("ALL-FILES check error")
    
        # ── ① SAF picker (androidstorage4kivy) 우선 사용 ──────────
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(
                    callback=self.on_choose,
                    multiple=True,
                    mime_type="text/*"       # CSV 포함
                )
                return                      # 성공적으로 열었으면 여기서 끝
            except Exception as e:
                Logger.warning(f"SAF picker 실패 → filechooser로 fallback: {e}")
    
        # ── ② 일반 filechooser ─────────────────────────────────────
        try:
            filechooser.open_file(
                on_selection=self.on_choose,
                multiple=True,
                filters=[("CSV", "*.csv")],
                path=DOWNLOAD_DIR
            )
        except Exception as e:
            self.log(f"파일 선택기를 열 수 없습니다: {e}")

    def _goto_allfiles_permission(self):
        from jnius import autoclass
        Intent  = autoclass("android.content.Intent")
        Settings= autoclass("android.provider.Settings")
        Uri     = autoclass("android.net.Uri")
        act     = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(
            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))

    # ↳ ❶  on_choose 시그니처 수정
    def on_choose(self, sel, *_) :
        """
        sel : [path1, path2, ...]  또는 [uri1, uri2, ...]
        *_  : 버전별 추가 인자 무시
        """
        if not sel:                # 취소
            return
        paths = []
        for raw in sel[:3]:        # 최대 3개(X,Y,Z)까지
            real = uri_to_file(raw)
            if not real:
                self.log("❌ 복사 실패"); return
            paths.append(real)
    
        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg, daemon=True).start()


    #   CSV 1 ~ 2개 FFT → 2 Hz 대역 RMS‧Peak + ΔF 계산
    # ─────────────────────────────────────────────────────
    def _fft_bg(self):
        try:
            all_sets, ym = [], 0.0      # [[rms, pk] …], y축 최대치

            for path in self.paths:     # ────── 파일별 반복 ──────
                t, a = self._load_csv(path)
                if t is None:
                    raise ValueError(f"{os.path.basename(path)}: CSV parse failed")

                # ── ① FFT – 가속도 스펙트럼 ─────────────────────
                n   = len(a)
                dt  = (t[-1] - t[0]) / (n-1) if n > 1 else 0.01
                sig = (a - a.mean()) * np.hanning(n)

                raw    = np.fft.fft(sig)
                amp_a  = 2*np.abs(raw[:n//2])/(n*np.sqrt(2))       # m/s² RMS
                freq   = np.fft.fftfreq(n, d=dt)[:n//2]

                # ── ② 5 Hz 하이패스 + 50 Hz 로우패스 ────────────
                msel = (freq >= HPF_CUTOFF) & (freq <= 50)
                freq, amp_a = freq[msel], amp_a[msel]
                if freq.size == 0:
                    raise ValueError("No data in 5-50 Hz band")

                # ── ③ ACC ↔ VEL 변환 ---------------------------
                amp_lin, REF0 = acc_to_spec(freq, amp_a)   # lin = 가속도 or 속도

                # ── ④ 2 Hz 대역 RMS / Peak 계산  -------------- 
                rms_line, pk_line = [], []
                for lo in np.arange(HPF_CUTOFF, 50, BAND_HZ):
                    hi  = lo + BAND_HZ
                    sel = (freq >= lo) & (freq < hi)
                    if not np.any(sel):
                        continue

                    rms = np.sqrt(np.mean(amp_lin[sel]**2))
                    pk  = amp_lin[sel].max()

                    centre = (lo + hi) / 2
                    rms_line.append((centre,
                                     20*np.log10(max(rms, REF0*1e-4)/REF0)))
                    pk_line .append((centre,
                                     20*np.log10(max(pk , REF0*1e-4)/REF0)))

                # ── ⑤ 스무딩 (선택) ────────────────────────────
                # _fft_bg  안
                if len(rms_line) >= SMOOTH_N:
                    y_sm = smooth_y([y for _, y in rms_line])   # 동일
                    rms_line = list(zip([x for x, _ in rms_line], y_sm))

                # ── ⑥ 공진주파수 저장 ─────────────────────────
                loF, hiF = FN_BAND
                if rms_line:
                    c = np.array([x for x, _ in rms_line])
                    m = np.array([y for _, y in rms_line])
                    s = (c >= loF) & (c <= hiF)
                    if s.any():
                        self.last_fn = c[s][m[s].argmax()]

                # ── ⑦ 누적 및 y-축 최대값 갱신 ───────────────
                all_sets.append([rms_line, pk_line])
                ym = max(ym,
                         max(y for _, y in rms_line),
                         max(y for _, y in pk_line))

            # ────────────────────────────────────────────────
            #   그래프 갱신 + ΔF 계산/로그
            # ────────────────────────────────────────────────
            if len(all_sets) == 1:           # ① 단일 파일
                r, p = all_sets[0]
                Clock.schedule_once(lambda *_:
                        self.graph.update_graph([r, p], [], 50, ym))

            else:                            # ② 두 파일 비교
                (r1, p1), (r2, p2) = all_sets[:2]

                diff = [(x, abs(y1 - y2) + self.OFFSET_DB)
                        for (x, y1), (_, y2) in zip(r1, r2)]
                ym = max(ym, max(y for _, y in diff))

                Clock.schedule_once(lambda *_:
                        self.graph.update_graph([r1, p1, r2, p2],
                                                diff, 50, ym))

                fn1 = max(r1, key=lambda p: p[1])[0]
                fn2 = max(r2, key=lambda p: p[1])[0]
                Clock.schedule_once(lambda *_:
                        self.log(f"CSV ΔF = {abs(fn1-fn2):.2f} Hz "
                                 f"({fn1:.2f} → {fn2:.2f})"))
        
        except Exception as e:
            msg = f"FFT 오류: {e}"
            Clock.schedule_once(lambda *_: self.log(msg)) 

        finally:
            Clock.schedule_once(lambda *_: setattr(self.btn_run,
                                                   "disabled", False))

    # CSV → 시계열 배열 읽기
    def _load_csv(self, path: str):
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
                    if not (num_re.match(row[0].strip()) and num_re.match(row[1].strip())):
                        continue
                    t.append(float(row[0].replace(",", ".")))
                    a.append(float(row[1].replace(",", ".")))
            if len(a) < 2:
                return None, None
            return np.asarray(t,float), np.asarray(a,float)
        except Exception as e:
            Logger.error(f"CSV read err ({os.path.basename(path)}): {e}")
            return None, None


        
    # ─────────────────────────────────────────────────────
    #  CSV 하나를 읽어 5 ~ 50 Hz / 2 Hz 대역의
    #  RMS·Peak(dB) 라인과 y-축 최대값을 반환
    # ─────────────────────────────────────────────────────
    @staticmethod
    def csv_fft(path: str):
        num_re = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?$")
        try:
            # ── ① CSV → 시계열 로드 ─────────────────────────
            t, a = [], []
            with open(path, encoding="utf-8", errors="replace") as f:
                sample = f.read(1024);  f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
                except csv.Error:
                    dialect = csv.get_dialect("excel")

                for row in csv.reader(f, dialect):
                    if len(row) < 2:
                        continue
                    if not (num_re.match(row[0].strip()) and
                            num_re.match(row[1].strip())):
                        continue
                    t.append(float(row[0].replace(",", ".")))
                    a.append(float(row[1].replace(",", ".")))

            if len(a) < 2:
                raise ValueError("too few numeric rows")

            # ── ② FFT (가속도 스펙트럼) ─────────────────────
            n   = len(a)
            dt  = (t[-1] - t[0]) / (n-1) if n > 1 else 0.01
            sig = (np.asarray(a, float) - np.mean(a)) * np.hanning(n)



            raw   = np.fft.fft(sig)
            amp_a = 2*np.abs(raw[:n//2])/(n*np.sqrt(2))          # m/s² RMS
            freq  = np.fft.fftfreq(n, d=dt)[:n//2]

            # ── ③ 5 Hz HPF + 50 Hz LPF ─────────────────────
            sel = (freq >= HPF_CUTOFF) & (freq <= 50)
            freq, amp_a = freq[sel], amp_a[sel]
            if freq.size == 0:
                raise ValueError("no data in 5–50 Hz band")

            # ── ④ ACC ↔ VEL 변환 ---------------------------
            amp_lin, REF0 = acc_to_spec(freq, amp_a)  # lin unit & 0 dB ref

            # ── ⑤ 2 Hz 대역별 RMS / Peak(dB) -------------- 
            band_rms, band_pk = [], []
            for lo in np.arange(HPF_CUTOFF, 50, BAND_HZ):
                hi  = lo + BAND_HZ
                m   = (freq >= lo) & (freq < hi)
                if not m.any():
                    continue
                rms = np.sqrt(np.mean(amp_lin[m]**2))
                pk  = amp_lin[m].max()

                centre = (lo + hi) / 2
                band_rms.append((centre,
                                 20*np.log10(max(rms, REF0*1e-4)/REF0)))
                band_pk .append((centre,
                                 20*np.log10(max(pk , REF0*1e-4)/REF0)))

            # ── ⑥ RMS 스무딩(선택) ─────────────────────────
            if len(band_rms) >= SMOOTH_N:
                y_sm = smooth_y([y for _, y in band_rms])
                band_rms = list(zip([x for x, _ in band_rms], y_sm))

            ymax = max(max(y for _, y in band_rms),
                       max(y for _, y in band_pk))
            return band_rms, band_pk, 50, ymax

        except Exception as e:
            Logger.error(f"FFT csv err ({os.path.basename(path)}): {e}")
            return None, None, 0, 0

# ── 실행 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
