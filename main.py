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

'''
#  간단 그래프 위젯  –  고정 색상·굵기 / 피크·Δ(정상·고장) 표시 안정판
# ────────────────────────────────────────────────────────────────
class GraphWidget(Widget):
    """2 개의 FFT 곡선을 그리고, 각 곡선의 **최대 피크 주파수**를 그래프
    위에 표시한다."""
    pad_x, pad_y = 80, 30
    COLORS  = [(1, 0, 0),   # 첫 번째 CSV  → 빨간색
               (0, 1, 0)]   # 두 번째 CSV  → 녹  색
    LINE_W  = 2.4           # 선 굵기

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    # ---------- 외부 호출 ----------
    def update_graph(self, ds, df, xm, ym):
        # 데이터 유효성 1차 확인
        ds = ds or []
        ds = [p for p in ds if p]          # None 요소 제거
        if not ds:
            Logger.warning("Graph: empty dataset, skip draw")
            return
    
        # diff 역시 리스트가 아닐 경우 방지
        df = df if isinstance(df, list) else []
    
        # 0-division 가드
        self.max_x = max(1e-6, xm)
        self.max_y = max(1e-6, ym)
    
        self.datasets, self.diff = ds, df
        self.redraw()

    # ---------- 내부 도우미 ----------
    def _scale(self, pts):
        w, h = max(1, self.width-2*self.pad_x), max(1, self.height-2*self.pad_y)
        out = []
        for x, y in pts:
            out += [self.pad_x + x/self.max_x*w,
                    self.pad_y + y/self.max_y*h]
        return out

    def _grid(self):
        gx, gy = (self.width-2*self.pad_x)/10, (self.height-2*self.pad_y)/10
        Color(.6, .6, .6)
        for i in range(11):
            Line(points=[self.pad_x+i*gx, self.pad_y,
                         self.pad_x+i*gx, self.height-self.pad_y])
            Line(points=[self.pad_x, self.pad_y+i*gy,
                         self.width-self.pad_x, self.pad_y+i*gy])

    def _labels(self):
        # 기존 축 라벨 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축: 0–50 Hz, 10 Hz 간격
        for i in range(6):
            freq = 10 * i
            x = self.pad_x + i*(self.width-2*self.pad_x)/5 - 18
            lab = Label(text=f"{freq:d} Hz", size_hint=(None,None),
                        size=(55,20), pos=(x, self.pad_y-28))
            lab._axis = True
            self.add_widget(lab)

        # Y축(좌·우)
        for i in range(11):
            mag = self.max_y*i/10
            y   = self.pad_y + i*(self.height-2*self.pad_y)/10 - 8
            for x in (self.pad_x-68, self.width-self.pad_x+10):
                lab = Label(text=f"{mag:.1e}", size_hint=(None,None),
                            size=(65,20), pos=(x,y))
                lab._axis = True
                self.add_widget(lab)

    # ---------- 메인 그리기 ----------
    def redraw(self, *_):
        self.canvas.clear()

        # 이전 피크 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks = []   # [(fx, fy, sx, sy)]

        with self.canvas:
            self._grid()
            self._labels()

            # FFT 곡선
            for idx, pts in enumerate(self.datasets):
                scaled = self._scale(pts)
                if len(scaled) < 4:
                    continue                      # 점 2개 미만이면 skip
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=scaled, width=self.LINE_W)

                # 최고점
                fx, fy = max(pts, key=lambda p: p[1])
                sx, sy = self._scale([(fx, fy)])[0:2]
                peaks.append((fx, sx, sy))

            # 차이선(흰색) — 필요하면 사용
            if self.diff:
                diff_scaled = self._scale(self.diff)
                if len(diff_scaled) >= 4:
                    Color(1,1,1); Line(points=diff_scaled, width=self.LINE_W)

        # 피크 주파수 라벨
        for fx, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(90,22),
                        pos=(sx-30, sy+6))
            lbl._peak = True
            self.add_widget(lbl)
'''


class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0),(0,1,0),(0,0,1)]
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.5

    MAX_FREQ = 50  # X축 0–50Hz 고정

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets = []
        self.diff     = []
        self.max_x = self.max_y = 1
        self.bind(size=lambda *a: Clock.schedule_once(lambda *_: self.redraw(), 0))

    def update_graph(self, ds, df, xm, ym):
        try:
            self.max_x = float(self.MAX_FREQ)
            self.max_y = max(1e-6, float(ym))
            self.datasets = [seq for seq in (ds or []) if seq]
            self.diff     = df or []
            Clock.schedule_once(lambda *_: self.redraw(), 0)
        except Exception as e:
            _dump_crash(f"update_graph error: {e}\n{traceback.format_exc()}")

    def _scale(self, pts):
        w = self.width  - 2*self.PAD_X
        h = self.height - 2*self.PAD_Y
        out = []
        for x, y in pts:
            out.append(self.PAD_X + (x/self.max_x)*w)
            out.append(self.PAD_Y + (y/self.max_y)*h)
        return out

    def _grid(self):
        gx = (self.width-2*self.PAD_X)/10
        gy = (self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        # 이전 축 레이블만 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축: 0,10,…,50 Hz
        step_x = self.MAX_FREQ / 5
        for i in range(6):
            x_val = i * step_x
            x_pos = self.PAD_X + (self.width-2*self.PAD_X)*(i/5) - 20
            lbl = Label(
                text=f"{int(x_val)} Hz",
                size_hint=(None,None), size=(60,20),
                pos=(int(x_pos), int(self.PAD_Y-28))
            )
            lbl._axis = True
            self.add_widget(lbl)

        # Y축: 0%, 50%, 100% 위치에만 레이블
        for frac in (0.0, 0.5, 1.0):
            mag   = self.max_y * frac
            y_pos = self.PAD_Y + (self.height-2*self.PAD_Y)*frac - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(
                    text=f"{mag:.1e}",
                    size_hint=(None,None), size=(60,20),
                    pos=(int(x), int(y_pos))
                )
                lbl._axis = True
                self.add_widget(lbl)

    def redraw(self, *_):
        try:
            # 충분한 크기가 확보되지 않으면 건너뛰기
            if self.width <= 2 * self.PAD_X or self.height <= 2 * self.PAD_Y:
                return
    
            # ① 기존 축·피크 레이블만 제거
            for child in list(self.children):
                if getattr(child, "_axis", False) or getattr(child, "_peak", False):
                    self.remove_widget(child)
    
            # ② 캔버스 초기화
            self.canvas.clear()
    
            if not self.datasets:
                return
    
            peaks = []
            with self.canvas:
                # 그리드
                self._grid()
                # 축 레이블 (Y축은 0%, 50%, 100% 기준)
                self._labels()
    
                # 데이터 곡선 및 피크 위치 계산
                for idx, pts in enumerate(self.datasets):
                    if not pts:
                        continue
                    Color(*self.COLORS[idx % len(self.COLORS)])
                    Line(points=self._scale(pts), width=self.LINE_W)
    
                    # 피크 계산
                    fx, fy = max(pts, key=lambda p: p[1])
                    sx, sy = self._scale([(fx, fy)])[0:2]
                    peaks.append((fx, fy, sx, sy))
    
                # 차이선 (필요 시)
                if self.diff:
                    Color(*self.DIFF_CLR)
                    Line(points=self._scale(self.diff), width=self.LINE_W)
    
            # ③ 피크 라벨 추가
            for fx, fy, sx, sy in peaks:
                lbl = Label(
                    text=f"▲ {fx:.1f} Hz",
                    size_hint=(None, None),
                    size=(85, 22),
                    pos=(int(sx - 28), int(sy + 6))
                )
                lbl._peak = True
                self.add_widget(lbl)
    
            # ④ Δ 표시 (첫 두 곡선만)
            if len(peaks) >= 2:
                delta = abs(peaks[0][0] - peaks[1][0])
                bad   = delta > 1.5
                clr   = (1, 0, 0, 1) if bad else (0, 1, 0, 1)
                info = Label(
                    text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                    size_hint=(None, None),
                    size=(190, 24),
                    pos=(int(self.PAD_X), int(self.height - self.PAD_Y + 6)),
                    color=clr
                )
                info._peak = True
                self.add_widget(info)
    
        except Exception as e:
            # 예외 발생 시 파일에 기록하고 로그에도 출력
            import traceback
            _dump_crash(f"redraw error: {e}\n{traceback.format_exc()}")
    
# ── 메인 앱 ───────────────────────────────────────────────────────
class FFTApp(App):
    RT_WIN   = 256
    FIXED_DT = 1.0 / 60.0  # 샘플링 간격 고정 (예: 60Hz)
    MIN_FREQ = 1.0         # 피크 검색시 1Hz 미만 제거

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # … 생략 …
        self.rt_buf = {
            'x': deque(maxlen=self.RT_WIN),
            'y': deque(maxlen=self.RT_WIN),
            'z': deque(maxlen=self.RT_WIN),
        }

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
    def toggle_realtime(self, *_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
    
        if self.rt_on:
            try:
                accelerometer.enable()
            except (NotImplementedError, Exception) as e:
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
    
    # ---------- ③ FFT 백그라운드 ----------
    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
                # 버퍼가 가득 찰 때까지 대기
            try:
                time.sleep(0.5)
                if any(len(self.rt_buf[ax]) < self.RT_WIN for ax in ('x','y','z')):
                    continue
    
                datasets = []
                ymax = xmax = 0.0
    
                for axis in ('x','y','z'):
                    _, vals = zip(*self.rt_buf[axis])
                    sig = np.asarray(vals, dtype=float)
                    n   = self.RT_WIN
    
                    # 고정 dt 사용
                    dt = self.FIXED_DT
    
                    # FFT
                    freq = np.fft.fftfreq(n, d=dt)[:n//2]
                    amp  = np.abs(fft(sig))[:n//2]
    
                    # 1–50Hz만 사용
                    mask = (freq <= self.graph.MAX_FREQ) & (freq >= self.MIN_FREQ)
                    freq = freq[mask]
                    smooth = np.convolve(amp[mask], np.ones(8)/8, 'same')
    
                    datasets.append(list(zip(freq, smooth)))
                    ymax = max(ymax, smooth.max() if len(smooth) else 0)
                    xmax = max(xmax, freq[-1] if len(freq) else 0)
    
                # 그래프 업데이트
                Clock.schedule_once(lambda *_:
                    self.graph.update_graph(datasets, [], xmax, ymax))

            except Exception as e:
                # 백그라운드 스레드의 예외도 로그에 남김
                _dump_crash(f"_rt_fft_loop error: {e}\n{traceback.format_exc()}")
                # 에러가 터져도 루프를 멈추지 않고 계속 시도할 수 있습니다.
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
        res=[]
        for p in self.paths:
            pts,xm,ym = self.csv_fft(p)
            if pts is None:
                self.log("CSV parse err"); return
            res.append((pts,xm,ym))

        if len(res)==1:
            pts,xm,ym = res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], xm, ym))
        else:
            (f1,x1,y1), (f2,x2,y2) = res
            diff=[(f1[i][0], abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2], diff, xm, ym))
        Clock.schedule_once(lambda *_:
            setattr(self.btn_run,"disabled",False))

    # ── CSV → FFT ────────────────────────────────────────────
    @staticmethod
    def csv_fft(path: str):
        try:
            t,a=[],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try:
                        t.append(float(r[0])); a.append(float(r[1]))
                    except Exception:
                        pass
            if len(a)<2:
                raise ValueError("too few samples")
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]
            s=np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f,s)), 50, s.max()
        except Exception as e:
            Logger.error(f"FFT err {e}")
            return None,0,0

# ── 실행 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
