"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판
"""

# ── 표준 및 3rd-party ───────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import numpy as np

# ── 최상단 import 부분 -------------
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False


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


#오디오 활용
#from jnius import autoclass, cast
# 1) 맨 위쪽 ─ Android 용 import 는 조건부로!
if ANDROID:
    from jnius import autoclass, cast, jarray        # ← 여기에만
    AudioRecord   = autoclass('android.media.AudioRecord')
    AudioFormat   = autoclass('android.media.AudioFormat')
    MediaRecorder = autoclass('android.media.MediaRecorder')
    ShortBuffer   = autoclass('java.nio.ShortBuffer')
else:
    # PC 빌드-타임에는 dummy 로 채워 두면 컴파일만 통과함
    autoclass = cast = jarray = lambda *a, **k: None
    AudioRecord = AudioFormat = MediaRecorder = ShortBuffer = None
    
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
            RECORD_AUDIO = ""
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
        # ① 기존 축 라벨 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)
    
        # ② X축 라벨 (max_x 크기에 따라 간격 조절)
        if   self.max_x <=  60: step = 10
        elif self.max_x <= 600: step = 100
        else:                   step = 300            # 0-1500 Hz
    
        nx = int(self.max_x // step) + 1
        for i in range(nx):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(nx-1) - 20
            lbl = Label(text=f"{i*step:d} Hz",
                        size_hint=(None,None), size=(60,20),
                        pos=(x, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)
    
        # ③ Y축 라벨 (좌·우, 지수 표기)
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

        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE, Permission.RECORD_AUDIO]
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



    SAMPLE_RATE   = 44100
    MIC_BUF_FRMS  = 1024          # 한 번에 읽어 올 프레임 수
    MIC_MAX_HZ    = 1500

    def _mic_start(self):
        """AudioRecord 열고 FFT 소비 스레드 기동"""
        cfg_ch  = AudioFormat.CHANNEL_IN_MONO
        cfg_fmt = AudioFormat.ENCODING_PCM_16BIT
        min_buf = AudioRecord.getMinBufferSize(self.SAMPLE_RATE,
                                               cfg_ch, cfg_fmt)
        buf_sz  = max(min_buf, self.MIC_BUF_FRMS*2)   # short = 2바이트

        self._j_rec = AudioRecord(MediaRecorder.AudioSource.MIC,
                                  self.SAMPLE_RATE,
                                  cfg_ch, cfg_fmt, buf_sz)
        self._j_rec.startRecording()

        self._mic_ring = deque(maxlen=4096)           # Python 측 버퍼
        self._mic_on   = True
        threading.Thread(target=self._mic_loop, daemon=True).start()

    def _mic_stop(self):
        self._mic_on = False
        try:
            self._j_rec.stop(); self._j_rec.release()
        except Exception:
            pass

    def _mic_loop(self):
        """백그라운드 – Java → numpy → FFT → 그래프"""
        # Java short[] 를 한 번만 생성해서 재사용
        j_short_arr = autoclass('jarray')('short')(self.MIC_BUF_FRMS)
        np_int16    = np.empty(self.MIC_BUF_FRMS, dtype=np.int16)

        while self._mic_on:
            # AudioRecord.read(short[] data, int offset, int size)
            read = self._j_rec.read(j_short_arr, 0, self.MIC_BUF_FRMS)
            if read <= 0:
                continue

            # Java short[] → numpy 배열
            ShortBuffer.wrap(j_short_arr).get(np_int16, 0, read)
            self._mic_ring.extend(np_int16[:read])

            if len(self._mic_ring) < 2048:
                continue          # 샘플 부족 → 계속 읽기

            # -------- FFT --------
            sig = np.array(self._mic_ring, dtype=float) / 32768.0
            self._mic_ring.clear()

            sig -= sig.mean()
            sig *= np.hanning(len(sig))
            n    = len(sig);  dt = 1.0 / self.SAMPLE_RATE
            f    = np.fft.fftfreq(n, dt)[:n//2]
            v    = np.abs(fft(sig))[:n//2]
            m    = f <= self.MIC_MAX_HZ
            f, v = f[m], v[m]
            smooth = np.convolve(v, np.ones(16)/16, 'same')
            pts    = list(zip(f, smooth))

            ymax = smooth.max()
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], self.MIC_MAX_HZ, ymax))

    def toggle_mic(self, *_):
        if getattr(self, '_mic_on', False):
            self._mic_stop()
            self.btn_mic.text = "Mic FFT (OFF)"
        else:
            try:
                self._mic_start()
                self.btn_mic.text = "Mic FFT (ON)"
            except Exception as e:
                self.log(f"Mic start fail: {e}")
                self.btn_mic.text = "Mic FFT (OFF)"


    
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


        self.btn_mic = Button(text="Mic FFT (OFF)", size_hint=(1, .1),
                              on_press=self.toggle_mic,
                              disabled=not HAVE_SD)   # ← SD 없으면 버튼 비활성
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
