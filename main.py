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

# ---------- 사용자 조정값 ---------- #
BAND_HZ     = 2.0     # ❶ RMS 를 묶을 주파수 대역폭(Hz)
REF_MM_S    = 0.01    # ❷ 0 dB 기준 속도 [mm/s RMS]
PEAK_COLOR  = (1,1,1) # ❸ 선형 피크 라인(흰색)
# ----------------------------------- #

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


# ── 그래프 위젯 (기존 그대로) ───────────────────────────────────────
# ── 그래프 위젯 (Y축 고정 · 세미로그) ───────────────────────────────
# ── 그래프 위젯 (Y축 고정 · 세미로그 · 좌표 캐스팅) ───────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0), (0,0,1), PEAK_COLOR]   # ← 맨 뒤 추가
    DIFF_CLR = (0,0,1)
    LINE_W   = 2.5

    Y_TICKS = [0, 40, 80, 150]
    Y_MAX   = Y_TICKS[-1]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = 1
        self.bind(size=self.redraw)

    # ---------- 외부 호출 ----------
    def update_graph(self, ds, df, xm, ym):
        self.max_x  = max(1e-6, float(xm))          # ← float 캐스팅
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
        # ▶ 최대값 받아서 20 dB 간격으로 라운드
        top = max(20, ((int(ym) // 20) + 1) * 20)      # 23 → 40, 67 → 80, …
        self.Y_TICKS = list(range(0, top + 1, 20))
        self.Y_MAX   = self.Y_TICKS[-1]
        
        self.redraw()

    def y_pos(self, v: float) -> float:
        """
        0-40 : 하단 40 %
        40-80: 40~70 %
        80-150: 70~100 %
        """
        h   = self.height - 2*self.PAD_Y
        v   = max(0.0, min(v, self.Y_MAX))

        if v <= 40:
            frac = 0.40 * (v / 40)
        elif v <= 80:
            frac = 0.40 + 0.30 * ((v - 40) / 40)
        else:          # 80-150
            frac = 0.70 + 0.30 * ((v - 80) / 70)

        return self.PAD_Y + frac * h

    
    # ---------- 좌표 변환 ----------
    def _scale(self, pts):
        """
        (주파수[Hz], dB) 튜플 리스트 →  [x1, y1, x2, y2, …]  로 변환
        """
        w = float(self.width  - 2*self.PAD_X)

        out = []
        for x, y in pts:
            sx = self.PAD_X + (float(x) / self.max_x) * w   # X-축 선형
            sy = self.y_pos(float(y))                       # Y-축 3-구간 압축
            out += [sx, sy]

        return out     

    # ---------- 그리드 ----------
    def _grid(self):
        gx = float(self.width - 2*self.PAD_X)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
        for v in self.Y_TICKS:
            y = self._scale([(0,v)])[1]
            Line(points=[self.PAD_X, y,
                         self.width-self.PAD_X, y])

    # ---------- 축 라벨 ----------
    def _labels(self):
        # 이전 축 라벨 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축
        for i in range(11):
            x = float(self.PAD_X + i*(self.width-2*self.PAD_X)/10 - 20)
            lbl = Label(text=f"{int(self.max_x*i/10)} Hz",
                        size_hint=(None,None), size=(60,20),
                        pos=(x, float(self.PAD_Y-28)))
            lbl._axis = True
            self.add_widget(lbl)

        # Y축
        for v in self.Y_TICKS:
            y = float(self._scale([(0,v)])[1] - 8)
            for x_pos in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{v}",
                            size_hint=(None,None), size=(60,20),
                            pos=(float(x_pos), y))
                lbl._axis = True
                self.add_widget(lbl)

    # ---------- 메인 그리기 ----------
    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks = []
        with self.canvas:
            self._grid(); self._labels()

            for idx, pts in enumerate(self.datasets):
                if not pts: continue
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                try:
                    fx, fy = max(pts, key=lambda p: p[1])
                    sx, sy = self._scale([(fx, fy)])[0:2]
                    peaks.append((fx, fy, sx, sy))
                except ValueError:
                    continue

            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(85,22),
                        pos=(float(sx-28), float(sy+6)))
            lbl._peak = True
            self.add_widget(lbl)

        # Δ 표시
        if len(peaks) >= 2:
            delta = abs(peaks[0][0] - peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            info  = Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                          size_hint=(None,None), size=(190,24),
                          pos=(float(self.PAD_X),
                               float(self.height-self.PAD_Y+6)),
                          color=clr)
            info._peak = True
            self.add_widget(info)
# ── 메인 앱 ────────────────────────────────────────────────────────
class FFTApp(App):
    REC_DURATION = 30.0          # 기록 길이(초)
    OFFSET_DB = 20 
    def __init__(self, **kw):
        super().__init__(**kw)
        # 실시간 FFT
        self.rt_on = False
        self.rt_buf = {ax: deque(maxlen=256) for ax in ('x','y','z')}
        # 30 초 기록
        self.rec_on = False
        self.rec_start = 0.0
        self.rec_files = {}

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
        self.label.text="Recording 0/30 s …"
        Clock.schedule_interval(self._record_poll, 0.02)

    def _record_poll(self, dt):
        if not self.rec_on: return False
        now=time.time()
        elapsed=now-self.rec_start
        try: ax,ay,az = accelerometer.acceleration
        except Exception as e:
            Logger.warning(f"acc read fail: {e}")
            ax=ay=az=None
        if None not in (ax,ay,az):
            t=elapsed
            for ax_name,val in (('x',ax),('y',ay),('z',az)):
                csv.writer(self.rec_files[ax_name]).writerow([t,val])
        if int(elapsed*2)%1==0:
            self.label.text=f"Recording {elapsed:4.1f}/30 s …"
        if elapsed>=self.REC_DURATION:
            self._stop_recording(); return False
        return True

    def _stop_recording(self):
        for f in self.rec_files.values():
            try: f.close()
            except Exception: pass
        self.rec_files.clear()
        self.rec_on=False
        self.btn_rec.disabled=False
        if not self.rt_on:
            try: accelerometer.disable()
            except Exception: pass
        self.log("✅ 30 초 기록 완료!")

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

    def _poll_accel(self, dt):
        if not self.rt_on: return False
        try:
            ax,ay,az = accelerometer.acceleration
            if None in (ax,ay,az): return
            now=time.time()
            self.rt_buf['x'].append((now,abs(ax)))
            self.rt_buf['y'].append((now,abs(ay)))
            self.rt_buf['z'].append((now,abs(az)))
        except Exception as e:
            Logger.warning(f"acc read fail: {e}")

    # ─────────────────────────────────────────────────────
    #  실시간 FFT 루프 – 2 Hz 대역별 ①RMS + ②피크(dB) 표시
    # ─────────────────────────────────────────────────────
    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
    
            # 샘플 수 부족 시 continue
            if any(len(self.rt_buf[ax]) < 64 for ax in ('x', 'y', 'z')):
                continue
    
            datasets = []          # 그래프에 그릴 모든 라인
            ymax     = 0
            xmax     = 50          # 항상 0-50 Hz
    
            for axis in ('x', 'y', 'z'):
                # ── ① 신호 가져오기 ─────────────────────
                ts, val = zip(*self.rt_buf[axis])
                sig     = np.asarray(val, float)
                n       = len(sig)
    
                dt  = (ts[-1] - ts[0]) / (n - 1) if n > 1 else 0.01
                sig = (sig - sig.mean()) * np.hanning(n)
    
                # ── ② 가속도 → 속도(mm/s RMS) 스펙트럼 ─
                raw    = np.fft.fft(sig)
                amp_a  = 2*np.abs(raw[:n//2])/(n*np.sqrt(2))        # m/s² RMS
                freq   = np.fft.fftfreq(n, d=dt)[:n//2]
    
                sel            = freq <= 50
                freq, amp_a    = freq[sel], amp_a[sel]
    
                f_nz  = np.where(freq < 1e-6, 1e-6, freq)
                amp_v = amp_a/(2*np.pi*f_nz)*1e3                    # mm/s RMS
    
                # ── ③ 2 Hz 대역별  RMS(dB) + 피크(dB) ──
                band_rms = []
                band_pk  = []
                for lo in np.arange(2, 50, BAND_HZ):
                    hi  = lo + BAND_HZ
                    s   = (freq >= lo) & (freq < hi)
                    if not np.any(s):
                        continue
    
                    # RMS
                    rms  = np.sqrt(np.mean(amp_v[s]**2))
                    db_r = 20*np.log10(max(rms, REF_MM_S*1e-4)/REF_MM_S)
                    band_rms.append(((lo+hi)/2, db_r))
    
                    # 피크
                    pk   = amp_v[s].max()
                    db_p = 20*np.log10(max(pk, REF_MM_S*1e-4)/REF_MM_S)
                    band_pk.append(((lo+hi)/2, db_p))
    
                # 살짝 스무딩
                if len(band_rms) > 2:
                    y = np.convolve([y for _, y in band_rms], np.ones(3)/3, "same")
                    band_rms = list(zip([x for x, _ in band_rms], y))
    
                # ── ④ 그래프용 데이터 push ─────────────────
                datasets.append(band_rms)   # 색선
                datasets.append(band_pk)    # 흰선
    
                ymax = max(ymax,
                           max(y for _, y in band_rms),
                           max(y for _, y in band_pk))
    
            # ── ⑤ UI 스레드로 그리기 요청 ─────────────────
            Clock.schedule_once(
                lambda *_: self.graph.update_graph(datasets, [], xmax, ymax)
            )

    # ── UI 구성 ───────────────────────────────────────────────
    def build(self):
        root = BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.label   = Label(text="Pick 1 or 2 CSV files",size_hint=(1,.10))
        self.btn_sel = Button(text="Select CSV",disabled=True,size_hint=(1,.08),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",disabled=True,size_hint=(1,.08),
                              on_press=self.run_fft)
        # ★ 30 초 기록 버튼
        self.btn_rec = Button(text="Record 30 s",disabled=True,size_hint=(1,.08),
                              on_press=self.start_recording)

        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(self.btn_rec)

        self.btn_rt  = Button(text="Realtime FFT (OFF)",size_hint=(1,.08),
                              on_press=self.toggle_realtime)
        root.add_widget(self.btn_rt)

        self.graph = GraphWidget(size_hint=(1,0.55)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ── CSV 선택 & FFT 실행 (기존) ───────────────────────────────
    def open_chooser(self,*_):
        if ANDROID and ANDROID_API>=30:
            try:
                from jnius import autoclass
                Env=autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():
                    mv=ModalView(size_hint=(.8,.35))
                    box=BoxLayout(orientation='vertical',spacing=10,padding=10)
                    box.add_widget(Label(
                        text="⚠️ CSV 파일에 접근하려면\n'모든 파일' 권한이 필요합니다.",
                        halign="center"))
                    box.add_widget(Button(text="권한 설정으로 이동",
                                          size_hint=(1,.4),
                                          on_press=lambda *_:(
                                              mv.dismiss(),
                                              self._goto_allfiles_permission())))
                    mv.add_widget(box); mv.open(); return
            except Exception:
                Logger.exception("ALL-FILES check 오류")
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True,mime_type="text/*")
                return
            except Exception as e:
                Logger.exception("SAF picker fail"); self.log(f"SAF 오류: {e}")
        try:
            filechooser.open_file(on_selection=self.on_choose,multiple=True,
                                  filters=[("CSV","*.csv")],native=False,
                                  path=DOWNLOAD_DIR)
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

    def on_choose(self, sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real=uri_to_file(raw)
            if not real:
                self.log("❌ 복사 실패"); return
            paths.append(real)
        self.paths=paths
        self.label.text=" · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled=False

    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg, daemon=True).start()


    # ─────────────────────────────────────────────────────
    #  CSV 파일 FFT : 2 Hz 대역 RMS(dB) & 피크(dB) + 차이선
    # ─────────────────────────────────────────────────────
    def _fft_bg(self):
        try:
            # 1) 각 CSV → 두 개의 라인(RMS·피크) 생성
            all_sets = []                  # [[rms, peak], …]
            ym = 0.0                       # y-축 최대치

            for path in self.paths:
                t, a = self._load_csv(path)          # (시간[], 가속도[])
                if t is None:
                    raise ValueError(f"{os.path.basename(path)}: CSV parse failed")

                # ── FFT → 속도(mm/s RMS) 스펙트럼 ──
                n   = len(a)
                dt  = (t[-1] - t[0]) / (n - 1) if n > 1 else 0.01
                sig = (a - a.mean()) * np.hanning(n)

                raw   = np.fft.fft(sig)
                freq  = np.fft.fftfreq(n, d=dt)[:n // 2]
                amp_a = 2 * np.abs(raw[:n // 2]) / (n * np.sqrt(2))  # m/s² RMS

                sel = freq <= 50
                freq, amp_a = freq[sel], amp_a[sel]

                amp_v = amp_a / (2 * np.pi * np.where(freq < 1e-6, 1e-6, freq)) * 1e3  # mm/s RMS

                rms_line, pk_line = [], []
                for lo in np.arange(2, 50, BAND_HZ):
                    hi = lo + BAND_HZ
                    m  = (freq >= lo) & (freq < hi)
                    if not m.any():
                        continue

                    rms = np.sqrt(np.mean(amp_v[m] ** 2))
                    pk  = amp_v[m].max()

                    db_r = 20 * np.log10(max(rms, REF_MM_S * 1e-4) / REF_MM_S)
                    db_p = 20 * np.log10(max(pk,  REF_MM_S * 1e-4) / REF_MM_S)

                    centre = (lo + hi) / 2
                    rms_line.append((centre, db_r))
                    pk_line.append((centre, db_p))

                # 살짝 스무딩(RMS 라인만)
                if len(rms_line) > 2:
                    y_sm = np.convolve([y for _, y in rms_line], np.ones(3) / 3, mode="same")
                    rms_line = list(zip([x for x, _ in rms_line], y_sm))

                all_sets.append([rms_line, pk_line])
                ym = max(ym,
                         max(y for _, y in rms_line),
                         max(y for _, y in pk_line))

            # 2) 그래프 그리기 데이터 구성
            if len(all_sets) == 1:
                r, p = all_sets[0]
                Clock.schedule_once(lambda *_:
                                    self.graph.update_graph([r , p], [], 50, ym))
            else:
                (r1, p1), (r2, p2) = all_sets[:2]

                diff = [(x, abs(y1 - y2) + self.OFFSET_DB)
                        for (x, y1), (_, y2) in zip(r1, r2)]

                ym = max(ym, max(y for _, y in diff))
                Clock.schedule_once(lambda *_:
                                    self.graph.update_graph([r1 , p1,
                                                             r2 , p2],
                                                            diff, 50, ym))

        except Exception as e:
            Clock.schedule_once(lambda *_: self.log(f"FFT 오류: {e}"))
        finally:
            # 버튼 활성화 복구
            Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))
            

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


        
    @staticmethod
    def csv_fft(path: str):
        # 수정 ― 지수부 (eE±) 허용
        num_re = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?$")
        try:
            t, a = [], []
            with open(path, encoding="utf-8", errors="replace") as f:
                sample = f.read(1024)
                f.seek(0)
            
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
                except csv.Error:
                    # ↙ 실패하면 그냥 'excel' 기본(콤마)로 진행
                    dialect = csv.get_dialect("excel")
                          
                rdr = csv.reader(f, dialect)
                for row in rdr:
                    if len(row) < 2:
                        continue                        # 열 2개 미만 skip
                    # --- 헤더/문자열 행 skip ---
                    if not (num_re.match(row[0].strip()) and
                            num_re.match(row[1].strip())):
                        continue
                    # 소수점 쉼표 → 점
                    t.append(float(row[0].replace(",", ".")))
                    a.append(float(row[1].replace(",", ".")))
    
            if len(a) < 2:
                raise ValueError("too few numeric rows")
    
            # ↓ 이하 FFT 부분은 그대로 …
    
            # ---------- 표본 주기 ----------
            dt = (t[-1] - t[0]) / (len(a) - 1)
            if dt <= 0:
                dt = 0.01                   # 100 Hz 가정(안전)
    

            # ---------- FFT(가속도 → 속도) ----------
            n      = len(a)
            sig    = (a - np.mean(a)) * np.hanning(n)
            raw    = np.fft.fft(sig)
            amp_a  = 2*np.abs(raw[:n//2])/(n*np.sqrt(2))          # m/s² RMS
            freq   = np.fft.fftfreq(n, d=dt)[:n//2]
            
            mask        = freq <= 50
            freq, amp_a = freq[mask], amp_a[mask]
            
            # ---- 가속도 → 속도(mm/s RMS) ----
            f_nz  = np.where(freq < 1e-6, 1e-6, freq)
            amp_v = amp_a/(2*np.pi*f_nz)*1e3                     # mm/s
            
            # ── ① 2 Hz 대역 RMS ─────────────────────
            # ---- 2 Hz 대역 RMS / Peak ---------------------------------
            band_db, band_pk = [], []
            for lo in np.arange(2, 50, BAND_HZ):
                hi  = lo + BAND_HZ
                sel = (freq >= lo) & (freq < hi)
                if not np.any(sel):
                    continue
                rms = np.sqrt(np.mean(amp_v[sel] ** 2))
                pk  = amp_v[sel].max()
            
                db  = 20*np.log10(max(rms, REF_MM_S*1e-4)/REF_MM_S)
                pkd = 20*np.log10(max(pk , REF_MM_S*1e-4)/REF_MM_S)
            
                centre = (lo + hi) / 2
                band_db.append((centre, db))
                band_pk.append((centre, pkd))
            
            # --- RMS 스무딩(3-point) ---
            if len(band_db) >= 3:
                y_smooth = np.convolve([y for _, y in band_db], np.ones(3)/3, mode="same")
                band_db  = list(zip([x for x, _ in band_db], y_smooth))
            
            ymax = max(max(y for _, y in band_db), max(y for _, y in band_pk))
            return band_db, band_pk, 50, ymax
            
    
        except Exception as e:
            Logger.error(f"FFT csv err ({os.path.basename(path)}): {e}")
            return None, 0, 0

# ── 실행 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
