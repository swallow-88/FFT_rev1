ㅁ미"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 안정판
+ 30 초 실시간 가속도 기록 (Downloads 폴더 저장 개선판)
"""
# ── 표준 & 3rd-party ────────────────────────────────────────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time
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
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS   = [(1,0,0), (0,1,0), (0,0,1)]
    DIFF_CLR = (1,1,1)
    LINE_W   = 2.5

    # ★ 고정 Y축 눈금값
    Y_TICKS  = [0, 5, 10, 20, 50]
    Y_MAX    = Y_TICKS[-1]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = 1          # X축은 여전히 자동
        self.bind(size=self.redraw)

    # ── 외부로부터 데이터 갱신 ─────────────────────────────
    def update_graph(self, ds, df, xm, _ym_ignored):
        self.max_x = max(1e-6, float(xm))   # X축만 갱신
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
        self.redraw()

    # ── 좌표 변환 (X는 선형 / Y는 세미로그 비율) ─────────────
    def _scale(self, pts):
        w = self.width  - 2*self.PAD_X
        h = self.height - 2*self.PAD_Y

        def y_pos(val: float) -> float:
            """
            0–5 : 전체 h의 0~40 %
            5–10: 40~60 %
            10–20:60~80 %
            20–50:80~100 %
            """
            v = max(0.0, min(val, self.Y_MAX))
            if   v <= 5:
                frac = 0.40 * (v/5)
            elif v <= 10:
                frac = 0.40 + 0.20*( (v-5)/5 )
            elif v <= 20:
                frac = 0.60 + 0.20*( (v-10)/10 )
            else:            # 20–50
                frac = 0.80 + 0.20*( (v-20)/30 )
            return self.PAD_Y + frac*h

        out=[]
        for x, y in pts:
            out += [ self.PAD_X + x/self.max_x*w,
                     y_pos(y) ]
        return out

    # ── 그리드선 --------------------------------------------------
    def _grid(self):
        gx = (self.width - 2*self.PAD_X)/10
        Color(.6,.6,.6)

        # 수직선 (X축 10등분)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])

        # 수평선 (고정 Y_TICKS 위치)
        for val in self.Y_TICKS:
            y = self._scale([(0,val)])[1]
            Line(points=[self.PAD_X, y,
                         self.width-self.PAD_X, y])

    # ── 축 라벨 ---------------------------------------------------
    def _labels(self):
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)

        # X축 0~max_x, 10등분
        n = 10
        for i in range(n+1):
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/n - 20
            lbl = Label(text=f"{int(self.max_x*i/n)} Hz",
                        size_hint=(None,None), size=(60,20),
                        pos=(x, self.PAD_Y-28))
            lbl._axis = True; self.add_widget(lbl)

        # 고정 Y축 라벨 (좌·우)
        for val in self.Y_TICKS:
            y = self._scale([(0,val)])[1] - 8
            for x_pos in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{val}",
                            size_hint=(None,None), size=(60,20),
                            pos=(x_pos, y))
                lbl._axis = True; self.add_widget(lbl)

    # ── 메인 그리기 루프 -----------------------------------------
    def redraw(self, *_):
        self.canvas.clear()
        # 이전 피크·Δ 라벨 제거
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)

        if not self.datasets:
            return

        peaks=[]
        with self.canvas:
            self._grid(); self._labels()

            # 곡선 및 피크
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

            # 차이선
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)

        # 피크 라벨
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz",
                        size_hint=(None,None), size=(85,22),
                        pos=(sx-28, sy+6))
            lbl._peak = True; self.add_widget(lbl)

        # Δ 표시
        if len(peaks) >= 2:
            delta = abs(peaks[0][0] - peaks[1][0])
            bad   = delta > 1.5
            clr   = (1,0,0,1) if bad else (0,1,0,1)
            info  = Label(text=f"Δ = {delta:.2f} Hz → {'고장' if bad else '정상'}",
                          size_hint=(None,None), size=(190,24),
                          pos=(self.PAD_X, self.height-self.PAD_Y+6),
                          color=clr)
            info._peak = True; self.add_widget(info)
# ── 메인 앱 ────────────────────────────────────────────────────────
class FFTApp(App):
    REC_DURATION = 30.0          # 기록 길이(초)

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

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[ax])<64 for ax in ('x','y','z')): continue
            datasets=[]; ymax=xmax=0.0
            for axis in ('x','y','z'):
                ts,val=zip(*self.rt_buf[axis]); sig=np.asarray(val,float)
                n=len(sig); dt=(ts[-1]-ts[0])/(n-1) if n>1 else 1/128.0
                sig -= sig.mean(); sig *= np.hanning(n)
                freq = np.fft.fftfreq(n,d=dt)[:n//2]
                amp  = np.abs(fft(sig))[:n//2]
                m=freq<=50; freq,amp=freq[m],amp[m]
                smooth = np.convolve(amp,np.ones(8)/8,'same')
                datasets.append(list(zip(freq,smooth)))
                ymax=max(ymax,smooth.max()); xmax=max(xmax,freq[-1])
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(datasets,[],xmax,ymax))

    # ── UI 구성 ───────────────────────────────────────────────
    def build(self):
        root = BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.label   = Label(text="Pick 1 or 2 CSV files",size_hint=(1,.1))
        self.btn_sel = Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                              on_press=self.run_fft)
        # ★ 30 초 기록 버튼
        self.btn_rec = Button(text="Record 30 s",disabled=True,size_hint=(1,.1),
                              on_press=self.start_recording)

        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(self.btn_rec)

        self.btn_rt  = Button(text="Realtime FFT (OFF)",size_hint=(1,.1),
                              on_press=self.toggle_realtime)
        root.add_widget(self.btn_rt)

        self.graph = GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
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
                self.graph.update_graph([pts],[],xm,ym))
        else:
            (f1,x1,y1),(f2,x2,y2) = res
            diff=[(f1[i][0],abs(f1[i][1]-f2[i][1]))
                  for i in range(min(len(f1),len(f2)))]
            xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2],diff,xm,ym))
        Clock.schedule_once(lambda *_:
            setattr(self.btn_run,"disabled",False))

    @staticmethod
    def csv_fft(path: str):
        try:
            t,a=[],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a)<2: raise ValueError("too few samples")
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]
            s=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,s)),50,s.max()
        except Exception as e:
            Logger.error(f"FFT err {e}")
            return None,0,0


# ── 실행 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    FFTApp().run()
