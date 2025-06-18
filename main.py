"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판 (2025-06-18)
"""

# ── 표준 및 3-rd-party ──────────────────────────────────────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time
from collections import deque

import numpy as np
from numpy.fft import fft

from plyer import accelerometer, filechooser               # SAF picker
from android.storage import app_storage_path

from kivy.app   import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color
from kivy.utils         import platform

# ── Android 전용 모듈 ───────────────────────────────────────────────
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
        from androidstorage4kivy import SharedStorage       # SAF 헬퍼
    except Exception:
        SharedStorage = None

    try:
        from android.permissions import (
            check_permission, request_permissions, Permission)
    except Exception:
        # permission recipe 가 없는 빌드용 더미
        check_permission = lambda *a, **kw: True
        request_permissions = lambda *a, **kw: None
        class _Dummy: pass
        Permission = _Dummy

    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ── 전역 예외 → /sdcard/fft_crash.log ──────────────────────────────
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

# ── SAF URI → 임시 파일 복사 ────────────────────────────────────────
def uri_to_file(u: str) -> str | None:
    if not u:
        return None
    if u.startswith("file://"):
        real = urllib.parse.unquote(u[7:])
        return real if os.path.exists(real) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    # SAF content:// → 내부 캐시에 복사
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                u, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

# ── 그래프 위젯 ─────────────────────────────────────────────────────
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0), (0,0,1)]
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.5

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    # 외부에서 호출
    def update_graph(self, ds, df, xm, ym, *, rt=False):
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
        self.max_x    = max(1e-6, float(xm))
        self.max_y    = max(1e-6, float(ym))
        self.redraw()

    # 좌표 변환
    def _scale(self, pts):
        w = self.width  - 2*self.PAD_X
        h = self.height - 2*self.PAD_Y
        return [c for x, y in pts
                  for c in (self.PAD_X + x/self.max_x*w,
                            self.PAD_Y + y/self.max_y*h)]

    # 눈금/라벨
    def _grid(self):
        gx = (self.width - 2*self.PAD_X) / 10
        gy = (self.height - 2*self.PAD_Y) / 10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축 0-50Hz 기준
        for i in range(6):
            freq = 10*i
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/5 - 18
            lbl = Label(text=f"{freq} Hz", size_hint=(None,None),
                        size=(55,20), pos=(x, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)

        # Y축
        for i in range(11):
            yval = self.max_y*i/10
            y = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for xx in (self.PAD_X-65, self.width-self.PAD_X+5):
                lbl = Label(text=f"{yval:.1e}", size_hint=(None,None),
                            size=(60,20), pos=(xx,y))
                lbl._axis = True
                self.add_widget(lbl)

    # 메인 그리기
    def redraw(self,*_):
        self.canvas.clear()
        # 피크/Δ 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)
        if not self.datasets:
            return

        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            # 곡선
            for idx, pts in enumerate(self.datasets):
                Color(*self.COLORS[idx % len(self.COLORS)])
                Line(points=self._scale(pts), width=self.LINE_W)
                try:
                    fx, fy = max(pts, key=lambda p: p[1])
                    sx, sy = self._scale([(fx, fy)])[0:2]
                    peaks.append((fx, sx, sy))
                except ValueError:
                    pass
            # 차이선
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz", size_hint=(None,None),
                        size=(90,22), pos=(sx-30, sy+6))
            lbl._peak = True
            self.add_widget(lbl)

        # Δ 주파수
        if len(peaks) >= 2:
            delta = abs(peaks[0][0] - peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            info = Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                         size_hint=(None,None), size=(190,24),
                         pos=(self.PAD_X, self.height-self.PAD_Y+6),
                         color=clr)
            info._peak=True
            self.add_widget(info)

# ── 메인 앱 ─────────────────────────────────────────────────────────
class FFTApp(App):
    RT_WIN = 256
    MIN_FREQ = 1.0

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rt_on = False
        self.rt_buf = {ax: deque(maxlen=self.RT_WIN) for ax in ('x','y','z')}
        self.prev_fft = None

    # ── 간단 로그
    def log(self,msg:str):
        Logger.info(msg); self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # ── 저장소 권한 확인 (SAF 있으면 생략)
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            return
        need=[Permission.READ_EXTERNAL_STORAGE] if hasattr(Permission,"READ_EXTERNAL_STORAGE") else []
        def _cb(p,g): self.btn_sel.disabled = not any(g)
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled=False
        else:
            request_permissions(need,_cb)

    # ── CSV / SAF 파일 선택
    def open_chooser(self,*_):
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                           multiple=True,
                                           mime_type="text/*")
                return
            except Exception as e:
                Logger.warning(f"SAF picker fail: {e}")
        # fallback
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True,
                              filters=[("CSV","*.csv")],
                              native=False,
                              path="/storage/emulated/0/Download")

    def on_choose(self, sel):
        if not sel: return
        paths=[]
        for raw in sel[:2]:
            real=uri_to_file(raw)
            Logger.info(f"{raw} → {real}")
            if not real:
                self.log("❌ 파일 복사 실패"); return
            paths.append(real)
        self.paths=paths
        self.label.text=" · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled=False

    # ── 실시간 센서
    def toggle_realtime(self,*_):
        self.rt_on = not self.rt_on
        self.btn_rt.text=f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try: accelerometer.enable()
            except Exception as e:
                self.log(f"센서 에러: {e}"); self.rt_on=False; return
            Clock.schedule_interval(self._poll_accel,0)
            threading.Thread(target=self._rt_fft_loop, daemon=True).start()
        else:
            try: accelerometer.disable()
            except Exception: pass

    def _poll_accel(self,dt):
        if not self.rt_on: return False
        try:
            ax,ay,az=accelerometer.acceleration
            if None in (ax,ay,az): return
            now=time.time()
            self.rt_buf['x'].append((now,abs(ax)))
            self.rt_buf['y'].append((now,abs(ay)))
            self.rt_buf['z'].append((now,abs(az)))
        except Exception as e:
            Logger.warning(f"accel read fail: {e}")

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[a])<64 for a in ('x','y','z')): continue
            data=[]; ymax=0
            for ax in ('x','y','z'):
                ts,val=zip(*self.rt_buf[ax]); n=len(val)
                sig=np.asarray(val,float); sig-=sig.mean(); sig*=np.hanning(n)
                dt=(ts[-1]-ts[0])/(n-1)
                f=np.fft.fftfreq(n,d=dt)[:n//2]; a=np.abs(fft(sig))[:n//2]
                m=f<=50; f,a=f[m],a[m]; s=np.convolve(a,np.ones(8)/8,'same')
                data.append(list(zip(f,s))); ymax=max(ymax,s.max())
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(data,[],50,ymax))
    # ── FFT 실행 (CSV) --------------------------------------------------
    def run_fft(self,*_):
        if not getattr(self,"paths",None): self.log("CSV 먼저 선택"); return
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res=[]
        for p in self.paths:
            pts,xm,ym=self.csv_fft(p)
            if pts is None: self.log("CSV parse err"); return
            res.append((pts,xm,ym))
        if len(res)==1:
            pts,xm,ym=res[0]
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts],[],xm,ym))
            self.prev_fft=[[pts]]
        else:
            (f1,x1,y1),(f2,x2,y2)=res
            diff=[(f1[i][0],abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2],diff,xm,ym))
            self.prev_fft=[[f1,f2]]
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    @staticmethod
    def csv_fft(path:str):
        try:
            t,a=[],[]
            with open(path,newline="") as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a)<2: raise ValueError("few samples")
            dt=(t[-1]-t[0])/(len(a)-1)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=(f>=1)&(f<=50); f,v=f[m],v[m]
            s=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,s)),50,s.max()
        except Exception as e:
            Logger.error(f"csv_fft err {e}")
            return None,0,0

    # ── 10 초 레코딩 & 저장 --------------------------------------------
    def record_10s(self,*_):
        if self.rt_on:
            self.log("⚠️ Realtime OFF 후 사용"); return
        self.btn_rec.disabled=True
        threading.Thread(target=self._record_10s_thread, daemon=True).start()

    def _record_10s_thread(self):
        try:
            accelerometer.enable()
            buf={'x':[],'y':[],'z':[]}; t0=time.time()
            while time.time()-t0<10:
                ax,ay,az=accelerometer.acceleration
                if None not in (ax,ay,az):
                    now=time.time()
                    buf['x'].append((now,ax))
                    buf['y'].append((now,ay))
                    buf['z'].append((now,az))
                time.sleep(0.005)
            accelerometer.disable()

            data=[]; ymax=0
            for ax in ('x','y','z'):
                ts,vals=zip(*buf[ax])
                sig=np.asarray(vals,float)*np.hanning(len(vals))
                dt=(ts[-1]-ts[0])/(len(vals)-1)
                f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]
                a=np.abs(fft(sig))[:len(sig)//2]
                m=(f>=1)&(f<=50); f,a=f[m],a[m]
                s=np.convolve(a,np.ones(8)/8,'same')
                data.append(list(zip(f,s))); ymax=max(ymax,s.max())

            Clock.schedule_once(lambda *_:
                self.graph.update_graph(data,[],50,ymax))

            # CSV 저장 → Downloads
            ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname=f"accel_{ts}.csv"
            internal=os.path.join(app_storage_path(), fname)
            with open(internal,"w") as fp:
                for i in range(len(buf['x'])):
                    fp.write(f"{buf['x'][i][0]},{buf['x'][i][1]},"
                             f"{buf['y'][i][1]},{buf['z'][i][1]}\n")
            if ANDROID and SharedStorage:
                SharedStorage().copy_to_shared(internal,fname)
                self.log(f"✅ 저장 완료 – Downloads/{fname}")
            else:
                self.log(f"✅ 저장 완료: {internal}")
        finally:
            Clock.schedule_once(lambda *_:
                setattr(self.btn_rec,"disabled",False))

    # ── UI ----------------------------------------------------------------
    def build(self):
        root=BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.label=Label(text="Pick 1 or 2 CSV files",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                            on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                            on_press=self.run_fft)
        root.add_widget(self.label); root.add_widget(self.btn_sel); root.add_widget(self.btn_run)
        root.add_widget(Button(text="EXIT",size_hint=(1,.1),on_press=self.stop))
        self.btn_rt=Button(text="Realtime FFT (OFF)",size_hint=(1,.1),
                           on_press=self.toggle_realtime)
        root.add_widget(self.btn_rt)
        self.btn_rec=Button(text="Record 10 s FFT",size_hint=(1,.1),
                            on_press=self.record_10s)
        root.add_widget(self.btn_rec)
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm,0)   # Select CSV 활성화
        return root

# ── 실행 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
