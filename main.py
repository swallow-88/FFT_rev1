# -*- coding: utf-8 -*-
"""
FFT CSV Viewer + Realtime Accelerometer FFT
(안드로이드 SAF·‘모든 파일’ 권한 대응 안정판)

 • CSV 1~2 개 비교 FFT
 • 휴대폰 3축 가속도 실시간 FFT (0-50 Hz)
"""

# ────────────────────── 표준/타사 ──────────────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time
from collections import deque
import numpy as np
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
from plyer               import filechooser, accelerometer

# (마이크 FFT용 — sounddevice 는 p4a 기본 레시피에 없음)
try:
    import sounddevice as sd            # 빌드에 recipe 추가 후 사용
    HAVE_SD = True
except Exception:
    HAVE_SD = False

# ──────────────────── Android 전용 모듈 ───────────────────
ANDROID = platform == "android"
toast = SharedStorage = check_permission = request_permissions = Permission = None
ANDROID_API = 0

if ANDROID:
    try:  from plyer import toast
    except Exception: toast = None

    try:  from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None

    try:
        from android.permissions import (
            check_permission, request_permissions, Permission)
    except Exception:
        # recipe 가 없으면 더미
        check_permission = lambda *a, **k: True
        request_permissions = lambda *a, **k: None
        class _P:
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
        Permission = _P

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ──────────────── 전역 예외 → /sdcard/fft_crash.log ───────────────
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

# ─────── SAF URI → 실제 파일(캐시) ───────
def uri_to_file(u: str) -> str | None:
    if not u: return None
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

# ═══════════════════ 그래프 위젯 ═══════════════════
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0), (0,0,1)]   # 빨/초/파
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.4

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets: list[list[tuple[float,float]]] = []
        self.diff    : list[tuple[float,float]]       = []
        self.max_x = self.max_y = 1.0
        self.bind(size=self.redraw)

    # ---------- 외부에서 데이터 주입 ----------
    def update_graph(self, ds, df, xm, ym):
        self.max_x = max(1e-6, float(xm))
        self.max_y = max(1e-6, float(ym))
        self.datasets = [p for p in (ds or []) if p]
        self.diff     = df or []
        self.redraw()

    # ---------- 내부 유틸 ----------
    def _scale(self, pts):
        w = max(1, self.width  - 2*self.PAD_X)
        h = max(1, self.height - 2*self.PAD_Y)
        out = []
        for x, y in pts:
            out += [self.PAD_X + x/self.max_x * w,
                    self.PAD_Y + y/self.max_y * h]
        return out

    def _grid(self):
        gx, gy = (self.width-2*self.PAD_X)/10, (self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        # 이전 축 라벨 제거
        for ch in list(self.children):
            if getattr(ch, "_axis", False):
                self.remove_widget(ch)

        # ---- X축 (max_x 에 따라 단계 자동) ----
        if   self.max_x <= 60:  step = 10
        elif self.max_x <= 600: step = 100
        else:                   step = 300           # ~1500 Hz
        n = int(self.max_x // step) + 1
        for i in range(n):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(n-1) - 22
            lbl = Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                        size=(60,20), pos=(x, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)

        # ---- Y축 (지수) ----
        for i in range(11):
            mag = self.max_y * i / 10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-70, self.width-self.PAD_X+10):
                lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                            size=(65,20), pos=(x,y))
                lbl._axis = True
                self.add_widget(lbl)

    # ---------- 메인 그리기 ----------
    def redraw(self,*_):
        self.canvas.clear()
        # 옛 피크·Δ 라벨 제거
        for ch in list(self.children):
            if getattr(ch, "_peak", False):
                self.remove_widget(ch)

        if not self.datasets: return
        peaks = []     # (fx,fy,sx,sy)

        with self.canvas:
            self._grid()
            self._labels()

            for idx, pts in enumerate(self.datasets):
                if len(pts) < 2: continue
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx, fy = max(pts, key=lambda p: p[1])
                sx, sy = self._scale([(fx, fy)])[0:2]
                peaks.append((fx, fy, sx, sy))

            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(90,22),
                        pos=(sx-30, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # 두 곡선 차이 Δ
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

# ═══════════════════ 메인 앱 ═══════════════════
class FFTApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        # 실시간 가속도
        self.rt_on  = False
        self.rt_buf = {ax: deque(maxlen=256) for ax in ('x','y','z')}
        # CSV 경로
        self.paths = []
        # 마이크
        self.mic_on = False
        self.mic_buf = deque(maxlen=4096)
        self.mic_stream = None

    # -------- 공통 로그/토스트 --------
    def log(self, msg):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # -------- 권한 체크 --------
    def _ask_perm(self,*_):
        # SAF 사용 시 파일권한 불필요
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            return

        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        MANAGE = getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)
        if MANAGE: need.append(MANAGE)
        if ANDROID_API >= 33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]

        def _cb(p,g):
            self.btn_sel.disabled = not any(g)
            if not any(g):
                self.log("저장소 권한 거부 → CSV 불가")
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need, _cb)

    # ============ UI ===============
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.label   = Label(text="Pick CSV (옵션) / 실시간 FFT ON", size_hint=(1,.1))
        self.btn_sel = Button(text="Select CSV", disabled=True, size_hint=(1,.1),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN", disabled=True, size_hint=(1,.1),
                              on_press=self.run_fft)
        self.btn_rt  = Button(text="Accel FFT (OFF)", size_hint=(1,.1),
                              on_press=self.toggle_realtime)
        mic_txt = "Mic FFT (OFF)" if HAVE_SD else "Mic FFT (미지원)"
        self.btn_mic = Button(text=mic_txt, size_hint=(1,.1),
                              on_press=self.toggle_mic,
                              disabled=not HAVE_SD)

        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(self.btn_rt)
        root.add_widget(self.btn_mic)
        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))

        self.graph = GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ============ CSV 파일 선택 ============
    def open_chooser(self,*_):
        # Android 11+ ‘모든-파일’ 안내
        if ANDROID and ANDROID_API >= 30:
            try:
                from jnius import autoclass
                Env = autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():
                    self._goto_allfiles_permission()
                    return
            except Exception: pass

        # SAF
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True, mime_type="text/*")
                return
            except Exception as e:
                Logger.error(f"SAF picker err: {e}")

        # 경로 chooser
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True,
                              filters=[("CSV","*.csv")],
                              native=False)

    def _goto_allfiles_permission(self):
        from jnius import autoclass
        Intent, Settings, Uri = (autoclass(x) for x in
            ("android.content.Intent",
             "android.provider.Settings",
             "android.net.Uri"))
        act = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))
        self.log("설정에서 ‘모든 파일’ 권한을 허용해 주세요")

    def on_choose(self, sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real = uri_to_file(raw)
            if not real:
                self.log("CSV 복사 실패"); return
            paths.append(real)
        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # ============ CSV FFT ============
    @staticmethod
    def csv_fft(path):
        try:
            t,a=[],[]
            with open(path, newline='') as f:
                for r in csv.reader(f):
                    try:  t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a) < 4: raise ValueError
            dt = (t[-1]-t[0]) / max(1, len(a)-1)
            f  = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v  = np.abs(fft(a))[:len(a)//2]
            m  = f <= 50
            f, v = f[m], v[m]
            s  = np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f, s)), 50, s.max()
        except Exception as e:
            Logger.error(f"csv_fft err: {e}")
            return None,0,0

    def run_fft(self,*_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res=[]
        for p in self.paths:
            pts,xm,ym = self.csv_fft(p)
            if pts is None: self.log("CSV parse err"); return
            res.append((pts,xm,ym))

        if len(res)==1:
            pts,xm,ym = res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], xm, ym))
        else:
            (f1,x1,y1),(f2,x2,y2)=res
            diff=[(f1[i][0], abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2], diff, xm, ym))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    # ============ 실시간 가속도 ============
    def toggle_realtime(self,*_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel, 0)
                threading.Thread(target=self._rt_fft_loop, daemon=True).start()
            except Exception as e:
                self.log(f"센서 ON 실패: {e}")
                self.rt_on=False; self.btn_rt.text="Accel FFT (OFF)"
        else:
            accelerometer.disable()

    def _poll_accel(self, dt):
        if not self.rt_on: return False
        try:
            ax,ay,az = accelerometer.acceleration
            if None in (ax,ay,az): return
            now = time.time()
            self.rt_buf['x'].append((now, abs(ax)))
            self.rt_buf['y'].append((now, abs(ay)))
            self.rt_buf['z'].append((now, abs(az)))
        except Exception: pass

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[ax])<64 for ax in ('x','y','z')):
                continue
            datasets=[]; ymax=0
            for ax in ('x','y','z'):
                ts,val = zip(*self.rt_buf[ax])
                sig=np.asarray(val,float); n=len(sig)
                dt = (ts[-1]-ts[0])/(n-1) if n>1 else 1/128.0
                sig -= sig.mean(); sig*=np.hanning(n)
                f = np.fft.fftfreq(n,d=dt)[:n//2]
                a = np.abs(fft(sig))[:n//2]
                m = f<=50; f,a = f[m],a[m]
                s = np.convolve(a, np.ones(8)/8,'same')
                datasets.append(list(zip(f,s)))
                ymax=max(ymax,s.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(datasets, [], 50, ymax))

    # ============ 마이크 (선택) ============
    def toggle_mic(self,*_):
        if not HAVE_SD:
            self.log("sounddevice 모듈이 없어 OFF")
            return
        self.mic_on = not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self._start_mic_stream()
            except Exception as e:
                self.log(f"Mic start err: {e}")
                self.mic_on=False; self.btn_mic.text="Mic FFT (OFF)"
        else:
            self._stop_mic_stream()

    def _start_mic_stream(self):
        self.mic_stream = sd.InputStream(samplerate=44100, channels=1,
                                         dtype='float32', blocksize=512,
                                         callback=self._on_mic_block)
        self.mic_stream.start()
        threading.Thread(target=self._mic_fft_loop, daemon=True).start()

    def _stop_mic_stream(self):
        try:
            self.mic_stream.stop(); self.mic_stream.close()
        except Exception: pass

    def _on_mic_block(self, in_data, *_):
        if self.mic_on:
            self.mic_buf.extend(in_data[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.25)
            if len(self.mic_buf)<2048: continue
            sig=np.array(self.mic_buf,float); self.mic_buf.clear()
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            n=len(sig); dt=1/44100.0
            f = np.fft.fftfreq(n,d=dt)[:n//2]
            a = np.abs(fft(sig))[:n//2]
            m = f<=1500; f,a=f[m],a[m]
            s = np.convolve(a, np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f,s))], [], 1500, s.max()))

# ─────────────── 실행 ───────────────
if __name__ == "__main__":
    FFTApp().run()
