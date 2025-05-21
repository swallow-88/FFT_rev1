"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판
"""

# ── 표준 및 3rd-party ───────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import numpy as np

import sounddevice as sd 

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
    COLORS   = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]   # 1번=빨, 2번=초록
    DIFF_CLR = (1, 1, 1)
    LINE_W   = 2.5

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    # ── 외부에서 호출 ─────────────────────────────────────────
    def update_graph(self, ds, df, xm, ym):
        # 0-division 방지
        self.max_x = max(1e-6, float(xm))
        self.max_y = max(1e-6, float(ym))

        # 빈 세트 걸러내기
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
        self.redraw()

    # ── 좌표 변환 ────────────────────────────────────────────
    # ─── 그래프 내부 좌표 변환 ─────────────────────────────
    def _scale(self, pts):
        w, h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [float(c)                                  # ← numpy → python float
                for x, y in pts
                for c in (self.PAD_X + x/self.max_x*w,
                           self.PAD_Y + y/self.max_y*h)]
    # ── 그리기 보조 ───────────────────────────────────────────
    def _grid(self):
        gx, gy = (self.width-2*self.PAD_X)/10, (self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        # 기존 축 레이블 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)
        '''
        # X축 (10 Hz 간격, 0-50 Hz)
        for i in range(6):
            x_lab = Label(text=f"{i*10:d} Hz",
                          size_hint=(None,None), size=(60,20),
                          pos=(self.PAD_X+i*(self.width-2*self.PAD_X)/5-20,
                               self.PAD_Y-28))
            x_lab._axis = True
            self.add_widget(x_lab)

        # Y축 (지수표기)
        for i in range(11):
            mag = self.max_y*i/10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
        '''
            '''
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                y_lab = Label(text=f"{mag:.1e}",
                              size_hint=(None,None), size=(60,20),
                              pos=(x, y))
                y_lab._axis = True
                self.add_widget(y_lab)
            '''

            # ── X축 : max_x 범위에 따라 간격 결정
            if   self.max_x <=  60: step = 10
            elif self.max_x <= 600: step = 100
            else:                   step = 300          # 1500 Hz 까지
    
            n = int(self.max_x // step) + 1
            for i in range(n):
                x = self.PAD_X + i*(self.width-2*self.PAD_X)/(n-1) - 20
                lbl = Label(text=f"{i*step:d} Hz",
                            size_hint=(None,None), size=(60,20),
                            pos=(x, self.PAD_Y-28))
                lbl._axis = True
                self.add_widget(lbl)
    
            # ── Y축 : 지수표기
            for i in range(11):
                mag = self.max_y * i / 10
                y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
                for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                    lbl = Label(text=f"{mag:.1e}",
                                size_hint=(None,None), size=(60,20),
                                pos=(x, y))
                    lbl._axis = True
                    self.add_widget(lbl)
        
    
    # ── 메인 그리기 ───────────────────────────────────────────
    def redraw(self,*_):
        self.canvas.clear()

        # 이전 피크·Δ 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks = []   # (fx, fy, sx, sy)

        with self.canvas:
            self._grid()
            self._labels()

            # ---------- 곡선 & 피크 ---------------------------------
            for idx, pts in enumerate(self.datasets):
                if not pts:
                    continue
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)

                # 안전하게 피크 계산
                try:
                    fx, fy = max(pts, key=lambda p: p[1])
                except ValueError:       # 빈 리스트
                    continue
                sx, sy = self._scale([(fx, fy)])[0:2]
                peaks.append((fx, fy, sx, sy))

            # ---------- 차이선 --------------------------------------
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # ---------- 피크 라벨 --------------------------------------
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(85,22),
                        pos=(sx-28, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # ---------- Δ 주파수 차 -----------------------------------
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
# ── 메인 앱 ───────────────────────────────────────────────────────
class FFTApp(App):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ── 실시간 가속도 FFT용 상태 ─────────────────────
        self.rt_on  = False               # 토글 상태
        self.rt_buf = {
            'x': deque(maxlen=256),
            'y': deque(maxlen=256),
            'z': deque(maxlen=256),
        }
        self.mic_on   = False
        self.mic_buf  = deque(maxlen=4096)
        self.mic_stream = None
    

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
            # 센서 시작 + 소비 스케줄
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel, 0)     # 프레임마다 센서 읽기
                threading.Thread(target=self._rt_fft_loop,
                                 daemon=True).start()
            except Exception as e:
                self.log(f"센서 활성화 실패: {e}")
                self.rt_on = False
                self.btn_rt.text = "Realtime FFT (OFF)"
        else:
            accelerometer.disable()


    # 마이크 토글
    # ─── ④·⑤·⑥번 : “마이크 관련” 메서드 묶음 ───────────
    def toggle_mic(self, *_):
        """UI 버튼 콜백 – ON/OFF 토글"""
        self.mic_on = not self.mic_on
        self.btn_mic.text = f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self._start_mic_stream()          # ④
            except Exception as e:
                self.log(f"마이크 시작 실패: {e}")
                self.mic_on = False
                self.btn_mic.text = "Mic FFT (OFF)"
        else:
            self._stop_mic_stream()               # ④

    # ④ 스트림 열고 닫기 -------------------------------------------------
    def _start_mic_stream(self):
        import sounddevice as sd                  # 파일 맨 위 import 권장
        self.mic_stream = sd.InputStream(
            samplerate=44100, channels=1, dtype='float32',
            blocksize=512,
            callback=self._on_mic_block)          # ⑤
        self.mic_stream.start()
        threading.Thread(target=self._mic_fft_loop,  # ⑥
                         daemon=True).start()

    def _stop_mic_stream(self):
        try:
            self.mic_stream.stop(); self.mic_stream.close()
        except Exception:
            pass

    # ⑤ 오디오 콜백 -----------------------------------------------------
    def _on_mic_block(self, in_data, frames, time_info, status):
        if not self.mic_on:
            return
        # in_data.shape == (frames, 1)
        self.mic_buf.extend(in_data[:, 0])

    # ⑥ FFT 백그라운드 루프 --------------------------------------------
    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.25)
            if len(self.mic_buf) < 2048:
                continue
            sig = np.array(self.mic_buf, dtype=float)
            self.mic_buf.clear()

            sig -= sig.mean(); sig *= np.hanning(len(sig))
            n   = len(sig);  dt = 1/44100.0
            freq = np.fft.fftfreq(n, d=dt)[:n//2]
            amp  = np.abs(fft(sig))[:n//2]

            mask = freq <= 1500
            freq, amp = freq[mask], amp[mask]
            smooth = np.convolve(amp, np.ones(16)/16, 'same')

            pts  = list(zip(freq, smooth))
            ymax = smooth.max()

            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], 1500, ymax))


  
    
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
        """
        0.5 s마다 X·Y·Z 각 축을 FFT 하고 그래프 3 개를 그려 준다.
        · 표본 간격(dt)을 실측으로 계산 → 유령 2 Hz 제거
        · DC 제거 + Hanning window 로 누수 감소
        · 0–50 Hz 구간만 부드럽게(smooth) 표시
        """
        while self.rt_on:
            time.sleep(0.5)

            # 64 샘플 미만이면 건너뜀
            if any(len(self.rt_buf[ax]) < 64 for ax in ('x', 'y', 'z')):
                continue

            datasets = []   # 그래프 3 개(X,Y,Z)
            ymax     = 0.0  # y축 최대
            xmax     = 0.0  # x축(샘플링 주파수 절반)

            for axis in ('x', 'y', 'z'):
                # ── 타임스탬프·데이터 분리 ──────────────────────
                ts, val = zip(*self.rt_buf[axis])      # 두 튜플로 분리
                sig = np.asarray(val, dtype=float)
                n   = len(sig)

                # ── 실측 dt 계산 ───────────────────────────────
                dt = (ts[-1] - ts[0]) / (n - 1) if n > 1 else 1/128.0

                # ── 전처리 : DC 제거 + 윈도잉 ───────────────────
                sig -= sig.mean()
                sig *= np.hanning(n)

                # ── FFT ───────────────────────────────────────
                freq = np.fft.fftfreq(n, d=dt)[:n//2]
                amp  = np.abs(fft(sig))[:n//2]

                mask = freq <= 50                # 0-50 Hz
                freq, amp = freq[mask], amp[mask]

                smooth = np.convolve(amp, np.ones(8)/8, 'same')

                datasets.append(list(zip(freq, smooth)))
                ymax = max(ymax, smooth.max())
                xmax = max(xmax, freq[-1])

            # ── 메인 스레드에서 그래프 갱신 ─────────────────────
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(datasets, [], xmax, ymax))
    

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


        self.btn_mic = Button(text="Mic FFT (OFF)", size_hint=(1,.1),
                              on_press=self.toggle_mic)
        root.add_widget(self.btn_mic)


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
