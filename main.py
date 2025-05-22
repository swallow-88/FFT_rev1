"""
FFT CSV / 가속도 / 마이크 FFT 뷰어 – Android SAF · AudioRecord 대응
"""
# ── 표준 모듈 ─────────────────────────────────────────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import time
from   collections import deque
import numpy as np
from   numpy.fft  import fft

# ── Kivy / Plyer ─────────────────────────────────────────────────────
from kivy.app      import App
from kivy.clock    import Clock
from kivy.logger   import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color
from kivy.utils         import platform
from plyer              import filechooser, accelerometer

# ── Android 전용 모듈 (조건부) ────────────────────────────────────────
ANDROID = platform == "android"

toast = SharedStorage = Permission = None
check_permission = request_permissions = lambda *_: True
ANDROID_API = 0

if ANDROID:
    # ① plyer · permissions
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
        class _P:  # ‘대체’ Permission 상수
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
            RECORD_AUDIO = MANAGE_EXTERNAL_STORAGE = ""
        Permission = _P

    # ② Pyjnius – AudioRecord 용
    try:
        from jnius import autoclass, jarray
        AudioRecord   = autoclass('android.media.AudioRecord')
        AudioFormat   = autoclass('android.media.AudioFormat')
        MediaRecorder = autoclass('android.media.MediaRecorder')
    except Exception as e:
        Logger.warning(f"Pyjnius import fail: {e}")
        AudioRecord = None           # 빌드 실패 방지용 더미
    try:
        from jnius import autoclass
        ANDROID_API = autoclass('android.os.Build$VERSION').SDK_INT
    except Exception:
        ANDROID_API = 0

# ─────────────────────────────────────────────────────────────────────
# 1. 공용 유틸 / 예외 덤프
# ─────────────────────────────────────────────────────────────────────
def _dump_crash(text: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" +
                     datetime.datetime.now().isoformat() + "\n" + text + "\n")
    except Exception:
        pass
    Logger.error(text)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)), size_hint=(.9,.9)).open())
sys.excepthook = _ex

# SAF URI → 임시 파일 경로 --------------------------------------------------
def uri_to_file(u: str) -> str | None:
    if not u:
        return None
    if u.startswith("file://"):
        p = urllib.parse.unquote(u[7:])
        return p if os.path.exists(p) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(u, uuid.uuid4().hex, False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

# ─────────────────────────────────────────────────────────────────────
# 2. 그래프 위젯  (CSV·가속도·마이크 모두 공유)
# ─────────────────────────────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS  = [(1,0,0), (0,1,0), (0,0,1)]   # R-G-B 순환
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.4

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, ds, df, xm, ym):
        self.max_x = max(1e-6, float(xm))
        self.max_y = max(1e-6, float(ym))
        self.datasets = [s for s in (ds or []) if s]
        self.diff     = df or []
        self.redraw()

    # - 내부 도우미 ----------------------------------------------------
    def _scale(self, pts):
        w, h = self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x, y in pts
                  for c in (self.PAD_X + x/self.max_x*w,
                            self.PAD_Y + y/self.max_y*h)]

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
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축 라벨 (max_x 범위별 간격)
        step = 10 if self.max_x <= 60 else (100 if self.max_x <= 600 else 300)
        nx   = int(self.max_x // step) + 1
        for i in range(nx):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(nx-1) - 20
            lbl = Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                        size=(60,20), pos=(x, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)

        # Y축 라벨 (지수)
        for i in range(11):
            mag = self.max_y * i / 10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                            size=(60,20), pos=(x, y))
                lbl._axis = True
                self.add_widget(lbl)

    # - 메인 그리기 -----------------------------------------------------
    def redraw(self,*_):
        self.canvas.clear()

        # 오래된 피크·Δ 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks = []
        with self.canvas:
            self._grid(); self._labels()

            # FFT 곡선
            for idx, pts in enumerate(self.datasets):
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx, fy = max(pts, key=lambda p:p[1])
                sx, sy = self._scale([(fx, fy)])[0:2]
                peaks.append((fx, sx, sy))

            # diff 선
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz", size_hint=(None,None),
                        size=(90,22), pos=(sx-30, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

# ─────────────────────────────────────────────────────────────────────
# 3. 메인 앱
# ─────────────────────────────────────────────────────────────────────
class FFTApp(App):
    SAMPLE_RATE  = 44100
    MIC_BUF_FRMS = 1024
    MIC_MAX_HZ   = 1500

    def __init__(self, **kw):
        super().__init__(**kw)
        # 가속도
        self.rt_on   = False
        self.rt_buf  = {k: deque(maxlen=256) for k in ('x','y','z')}
        # 마이크
        self._mic_active = False
        self._rec = None
        self._mic_ring = deque(maxlen=4096)

    # ── 작은 로그 + 토스트 ------------------------------------------
    def log(self, msg):
        Logger.info(msg); self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ── 저장소 + AUDIO 권한 -----------------------------------------
    def _ask_perm(self,*_):
        if not ANDROID:
            self.btn_sel.disabled = False; return

        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE,
                Permission.RECORD_AUDIO]
        MANAGE = getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None)
        if MANAGE: need.append(MANAGE)
        if ANDROID_API >= 33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]

        def _cb(p, g):
            ok = any(g)
            self.btn_sel.disabled = not ok
            if not ok: self.log("저장소 / 오디오 권한 거부")

        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need, _cb)

    # ── UI ----------------------------------------------------------
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.label = Label(text="Pick CSV or use sensors", size_hint=(1,.1))
        self.btn_sel = Button(text="Select CSV", size_hint=(1,.1),
                              disabled=True, on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN", size_hint=(1,.1),
                              disabled=True, on_press=self.run_fft)
        self.btn_rt  = Button(text="Realtime FFT (OFF)", size_hint=(1,.1),
                              on_press=self.toggle_rt)
        self.btn_mic = Button(text="Mic FFT (OFF)", size_hint=(1,.1),
                              on_press=self.toggle_mic)

        for w in (self.label, self.btn_sel, self.btn_run,
                  self.btn_rt, self.btn_mic,
                  Button(text="EXIT", size_hint=(1,.1), on_press=self.stop)):
            root.add_widget(w)

        self.graph = GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ── 파일 선택 ----------------------------------------------------
    def open_chooser(self,*_):
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True,
                              filters=[("CSV","*.csv")],
                              native=False)

    def on_choose(self, sel):
        if not sel: return
        self.paths = [uri_to_file(p) for p in sel[:2] if uri_to_file(p)]
        if not self.paths:
            self.log("파일 선택 실패"); return
        self.label.text = " · ".join(os.path.basename(p) for p in self.paths)
        self.btn_run.disabled = False

    # ── CSV FFT -----------------------------------------------------
    @staticmethod
    def csv_fft(path):
        try:
            t,a = [],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a)<2: raise ValueError
            dt  = (t[-1]-t[0])/len(a)
            f   = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v   = np.abs(fft(a))[:len(a)//2]
            m   = f<=50; f,v = f[m],v[m]
            s   = np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f,s)), 50, s.max()
        except Exception as e:
            Logger.error(f"CSV FFT err {e}"); return None,0,0

    def run_fft(self,*_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res=[]
        for p in self.paths:
            pts,xm,ym = self.csv_fft(p)
            if pts is None: return
            res.append((pts,xm,ym))
        if len(res)==1:
            pts,xm,ym = res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], xm, ym))
        else:
            (p1,x1,y1), (p2,x2,y2) = res
            diff=[(p1[i][0], abs(p1[i][1]-p2[i][1]))
                  for i in range(min(len(p1),len(p2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([p1,p2], diff, xm, ym))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    # ── 실시간 가속도 -------------------------------------------------
    def toggle_rt(self,*_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            accelerometer.enable()
            Clock.schedule_interval(self._poll_acc, 0)
            threading.Thread(target=self._rt_fft_loop, daemon=True).start()
        else:
            accelerometer.disable()

    def _poll_acc(self, dt):
        if not self.rt_on: return False
        ax,ay,az = accelerometer.acceleration
        if None in (ax,ay,az): return
        now = time.time()
        self.rt_buf['x'].append((now,abs(ax)))
        self.rt_buf['y'].append((now,abs(ay)))
        self.rt_buf['z'].append((now,abs(az)))

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[a])<64 for a in self.rt_buf): continue
            ds=[]; ymax=0
            for a in 'xyz':
                ts,val = zip(*self.rt_buf[a])
                sig = np.asarray(val,float); n=len(sig)
                dt = (ts[-1]-ts[0])/(n-1) if n>1 else 1/128
                sig -= sig.mean(); sig*=np.hanning(n)
                f = np.fft.fftfreq(n,dt)[:n//2]; v=np.abs(fft(sig))[:n//2]
                m=f<=50; f,v=f[m],v[m]
                s=np.convolve(v,np.ones(8)/8,'same')
                ds.append(list(zip(f,s))); ymax=max(ymax,s.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(ds, [], 50, ymax))

    # ── 마이크 FFT (AudioRecord) -------------------------------------
    def toggle_mic(self,*_):
        if self._mic_active:
            self._mic_stop()
            self.btn_mic.text="Mic FFT (OFF)"
        else:
            try:
                self._mic_start()
                self.btn_mic.text="Mic FFT (ON)"
            except Exception as e:
                self.log(f"Mic start fail: {e}")
                self.btn_mic.text="Mic FFT (OFF)"

    def _mic_start(self):
        if not ANDROID or AudioRecord is None:
            raise RuntimeError("AudioRecord unavailable")
        ch, fmt = AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        minbuf  = AudioRecord.getMinBufferSize(self.SAMPLE_RATE, ch, fmt)
        buf_sz  = max(minbuf, self.MIC_BUF_FRMS*2)   # short = 2 bytes
        self._rec = AudioRecord(MediaRecorder.AudioSource.MIC,
                                self.SAMPLE_RATE, ch, fmt, buf_sz)
        self._rec.startRecording()
        self._mic_active = True
        threading.Thread(target=self._mic_loop, daemon=True).start()

    def _mic_stop(self):
        self._mic_active = False
        try:
            self._rec.stop(); self._rec.release()
        except Exception: pass

    def _mic_loop(self):
        jshort_arr = jarray('h')(self.MIC_BUF_FRMS)   # Java short[]
        while self._mic_active:
            read = self._rec.read(jshort_arr, 0, self.MIC_BUF_FRMS)
            if read <= 0: continue
            np_int16 = np.frombuffer(jshort_arr.tobytes(),
                                     dtype=np.int16, count=read)
            self._mic_ring.extend(np_int16)
            if len(self._mic_ring) < 2048: continue
            sig = np.array(self._mic_ring, float) / 32768.0
            self._mic_ring.clear()
            sig -= sig.mean(); sig*=np.hanning(len(sig))
            n=len(sig); dt=1./self.SAMPLE_RATE
            f=np.fft.fftfreq(n,dt)[:n//2]; v=np.abs(fft(sig))[:n//2]
            m=f<=self.MIC_MAX_HZ; f,v=f[m],v[m]
            s=np.convolve(v,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f,s))],[],
                                         self.MIC_MAX_HZ, s.max()))

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
