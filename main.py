"""
FFT CSV / 가속도 / 마이크 뷰어 – Android SAF·권한 대응 통합판
2025-05-21
"""

# ──────────────────── 표준·3rd-party ────────────────────
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import time
from   collections import deque

import numpy as np
from   numpy.fft import fft

# ── 실시간 센서 ----------------------------------------------------------------
from plyer import accelerometer                 # 가속도
try:                                            # 마이크 (PC 테스트용)
    import sounddevice as sd
except Exception:
    sd = None                                   # p4a 레시피 없으면 None

# ── Kivy ────────────────────────────────────────────────────────────
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
from plyer               import filechooser      # SAF 실패 fallback

# ── Android 전용 모듈(존재할 때만) ───────────────────────────────
ANDROID = platform == "android"

toast = SharedStorage = Permission = None
check_permission = request_permissions = lambda *_: True
ANDROID_API = 0

if ANDROID:
    try:    from plyer import toast
    except Exception: toast = None

    try:    from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None

    try:
        from android.permissions import (
            check_permission, request_permissions, Permission)
    except Exception:
        class _P:                 # 더미
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
            MANAGE_EXTERNAL_STORAGE = ""
            RECORD_AUDIO = ""
        Permission = _P

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ── 공용 유틸 ──────────────────────────────────────────────────────
def _dump_crash(msg: str):
    """Android → /sdcard/fft_crash.log 에 크래시 스택 저장"""
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" +
                     datetime.datetime.now().isoformat() + "\n" + msg + "\n")
    except Exception:
        pass
    Logger.error(msg)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)), size_hint=(.9,.9)).open())
sys.excepthook = _ex

def uri_to_file(u: str) -> str | None:
    """SAF URI → 실제 파일(캐시) 경로로 변환"""
    if not u: return None
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

# ──────────────────── 그래프 위젯 ─────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0),(0,1,0),(0,0,1)]   # 빨·초·파 (3축/CSV 2개까지)
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.3

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, ds, df, xm, ym):
        self.datasets = [lst for lst in (ds or []) if lst]
        self.diff     = df or []
        self.max_x    = max(1e-6, float(xm))
        self.max_y    = max(1e-6, float(ym))
        self.redraw()

    # ---------- 내부 도우미 ----------
    def _scale(self, pts):
        w, h = max(1, self.width-2*self.PAD_X), max(1, self.height-2*self.PAD_Y)
        return [coord
                for x, y in pts
                for coord in (self.PAD_X + x/self.max_x*w,
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
        # 기존 축 레이블 제거
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축 : max_x 구간별 간격 자동
        step = 10 if self.max_x <= 60 else (100 if self.max_x <= 600 else 300)
        n    = int(self.max_x // step) + 1
        for i in range(n+1):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(n if n else 1)
            lbl = Label(text=f"{i*step:d} Hz", size_hint=(None,None),
                        size=(60,20), pos=(x-20, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)

        # Y축 : 0–max_y 를 10등분
        for i in range(11):
            mag = self.max_y * i / 10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                            size=(60,20), pos=(x, y))
                lbl._axis = True
                self.add_widget(lbl)

    # ---------- 메인 그리기 ----------
    def redraw(self,*_):
        self.canvas.clear()
        # 이전 피크·Δ 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets: return

        peaks = []          # [(fx, fy, sx, sy)]
        with self.canvas:
            self._grid(); self._labels()

            # 그래프 선 & 피크
            for idx, pts in enumerate(self.datasets):
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                fx, fy = max(pts, key=lambda p:p[1])
                sx, sy = self._scale([(fx, fy)])[0:2]
                peaks.append((fx, fy, sx, sy))

            # Δ 선 (CSV 2개 비교용)
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲{fx:.1f} Hz", size_hint=(None,None),
                        size=(80,22), pos=(sx-28, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # Δ 표시
        if len(peaks) >= 2:
            delta = abs(peaks[0][0]-peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            info  = Label(text=f"Δ={delta:.2f} Hz → {'고장' if bad else '정상'}",
                          size_hint=(None,None), size=(190,24),
                          pos=(self.PAD_X, self.height-self.PAD_Y+6),
                          color=clr)
            info._peak=True
            self.add_widget(info)

# ──────────────────── 메인 앱 ───────────────────────────────
class FFTApp(App):
    def __init__(self, **kw):
        super().__init__(**kw)
        # 가속도
        self.rt_on = False
        self.rt_buf = {a: deque(maxlen=256) for a in 'xyz'}
        # 마이크
        self.mic_on = False
        self.mic_buf = deque(maxlen=4096)
        self.mic_stream = None

    # ---------- 작은 로그 ----------
    def log(self, msg:str):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ---------- 권한 ----------
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            return
        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if getattr(Permission, "MANAGE_EXTERNAL_STORAGE", ""):   # Android 11+
            need.append(Permission.MANAGE_EXTERNAL_STORAGE)
        need.append(Permission.RECORD_AUDIO)                      # 마이크
        if ANDROID_API >= 33:
            need += [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]

        def _cb(_, grants):
            self.btn_sel.disabled = not any(grants)
            if not any(grants):
                self.log("저장소/오디오 권한 거부됨")

        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need, _cb)

    # ---------- 실시간 가속도 ----------
    def toggle_realtime(self, *_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel, 0)
                threading.Thread(target=self._rt_fft_loop,daemon=True).start()
            except Exception as e:
                self.log(f"Accel 시작 실패: {e}")
                self.toggle_realtime()
        else:
            accelerometer.disable()

    def _poll_accel(self, _dt):
        if not self.rt_on: return False
        try:
            ax, ay, az = accelerometer.acceleration
            if None in (ax, ay, az): return
            now=time.time()
            self.rt_buf['x'].append((now,abs(ax)))
            self.rt_buf['y'].append((now,abs(ay)))
            self.rt_buf['z'].append((now,abs(az)))
        except Exception as e:
            Logger.warning(f"accel read fail: {e}")

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[a])<64 for a in 'xyz'): continue
            datasets=[]; ymax=0
            for a in 'xyz':
                ts,val=zip(*self.rt_buf[a])
                sig=np.asarray(val,float); n=len(sig)
                dt=(ts[-1]-ts[0])/(n-1) if n>1 else 1/128
                sig-=sig.mean(); sig*=np.hanning(n)
                f=np.fft.fftfreq(n,d=dt)[:n//2]
                v=np.abs(fft(sig))[:n//2]
                m=f<=50; smooth=np.convolve(v[m],np.ones(8)/8,'same')
                datasets.append(list(zip(f[m],smooth)))
                ymax=max(ymax,smooth.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(datasets, [], 50, ymax))

    # ---------- 마이크 ----------
    def toggle_mic(self,*_):
        if sd is None:                        # 레시피 없으면 불가
            self.log("sounddevice 모듈이 없습니다")
            return
        self.mic_on = not self.mic_on
        self.btn_mic.text = f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self._start_mic()
            except Exception as e:
                self.log(f"Mic 시작 실패: {e}")
                self.toggle_mic()
        else:
            self._stop_mic()

    def _start_mic(self):
        self.mic_stream = sd.InputStream(
            samplerate=44100, channels=1, dtype='float32',
            blocksize=512, callback=self._on_mic_block)
        self.mic_stream.start()
        threading.Thread(target=self._mic_fft_loop,daemon=True).start()

    def _stop_mic(self):
        try: self.mic_stream.stop(); self.mic_stream.close()
        except Exception: pass

    def _on_mic_block(self, in_data, frames, *_):
        if self.mic_on:
            self.mic_buf.extend(in_data[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.25)
            if len(self.mic_buf)<2048: continue
            sig=np.array(self.mic_buf,float); self.mic_buf.clear()
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            n=len(sig); dt=1/44100
            f=np.fft.fftfreq(n,d=dt)[:n//2]
            v=np.abs(fft(sig))[:n//2]
            m=f<=1500; smooth=np.convolve(v[m],np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f[m],smooth))],[],1500,smooth.max()))

    # ---------- 파일 FFT ----------
    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg,daemon=True).start()

    def _fft_bg(self):
        res=[]
        for p in self.paths:
            pts,xm,ym=self.csv_fft(p)
            if pts is None:
                self.log("CSV parse err"); return
            res.append((pts,xm,ym))
        if len(res)==1:
            pts,xm,ym=res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts],[],xm,ym))
        else:
            (f1,x1,y1),(f2,x2,y2)=res
            diff=[(f1[i][0],abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2],diff,xm,ym))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    @staticmethod
    def csv_fft(path:str):
        try:
            t,a=[],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a)<2: raise ValueError
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]
            s=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,s)), 50, s.max()
        except Exception as e:
            Logger.error(f"FFT csv err {e}")
            return None,0,0

    # ---------- UI ----------
    def build(self):
        root=BoxLayout(orientation='vertical',padding=10,spacing=10)
        self.label=Label(text="Pick 1–2 CSV files",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                            on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                            on_press=self.run_fft)
        self.btn_rt =Button(text="Accel FFT (OFF)",size_hint=(1,.1),
                            on_press=self.toggle_realtime)
        self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.1),
                            on_press=self.toggle_mic)
        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(self.btn_rt)
        root.add_widget(self.btn_mic)
        root.add_widget(Button(text="EXIT",size_hint=(1,.1),on_press=self.stop))
        self.graph=GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

    # ---------- 파일 선택 ----------
    def open_chooser(self,*_):
        # Android 11+ “모든-파일” 권한 안내 생략 (기존 코드와 동일)
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True,mime_type="text/*")
                return
            except Exception as e:
                Logger.error(f"SAF picker fail {e}")
        filechooser.open_file(on_selection=self.on_choose,multiple=True,
                              filters=[("CSV","*.csv")],native=False)

    def on_choose(self, sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real=uri_to_file(raw)
            if not real:
                self.log("복사 실패"); return
            paths.append(real)
        self.paths=paths
        self.label.text=" · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled=False

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
