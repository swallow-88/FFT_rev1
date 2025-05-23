"""
FFT CSV Viewer – SAF + Android ‘모든-파일’ 권한 대응 + 실시간 가속도・마이크 FFT
"""

import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import numpy as np
import sounddevice as sd
from collections import deque
import queue, time

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

# ── Android 専用モジュール ---------------------------------------
ANDROID = platform == "android"
toast = None
SharedStorage = None
Permission = None
check_permission = request_permissions = None
ANDROID_API = 0

if ANDROID:
    try:
        from plyer import toast
    except:
        toast = None
    try:
        from androidstorage4kivy import SharedStorage
    except:
        SharedStorage = None
    try:
        from android.permissions import check_permission, request_permissions, Permission
    except:
        check_permission = lambda *a, **k: True
        request_permissions = lambda *a, **k: None
        class _P:
            READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = RECORD_AUDIO = ""
            READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
        Permission = _P
    try:
        from jnius import autoclass
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except:
        ANDROID_API = 0

# ── Crash ログを /sdcard/fft_crash.log に出力 ----------------------
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n" + "="*60 + "\n" +
                     datetime.datetime.now().isoformat() + "\n" + txt + "\n")
    except:
        pass
    Logger.error(txt)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et, ev, tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)), size_hint=(.9,.9)).open())
sys.excepthook = _ex

# ── SAF URI → キャッシュファイルパス変換 ---------------------------
def uri_to_file(u: str) -> str|None:
    if not u: return None
    if u.startswith("file://"):
        real = urllib.parse.unquote(u[7:])
        return real if os.path.exists(real) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(u, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
    return None

# ── グラフ描画ウィジェット -----------------------------------------
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS       = [(1,0,0),(0,1,0),(0,0,1)]
    DIFF_CLR     = (1,1,1)
    LINE_W       = 2.5

    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, ds, df, xm, ym):
        # division-by-zero guard
        self.max_x = max(1e-6, float(xm))
        self.max_y = max(1e-6, float(ym))
        # filter empty series
        self.datasets = [seq for seq in (ds or []) if seq]
        self.diff     = df or []
        self.redraw()

    def _scale(self, pts):
        w = self.width - 2*self.PAD_X
        h = self.height - 2*self.PAD_Y
        return [float(c) for x,y in pts
                    for c in (self.PAD_X + x/self.max_x*w,
                              self.PAD_Y + y/self.max_y*h)]

    def _grid(self):
        gx = (self.width  - 2*self.PAD_X)/10
        gy = (self.height - 2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx, self.PAD_Y,
                         self.PAD_X+i*gx, self.height-self.PAD_Y])
            Line(points=[self.PAD_X, self.PAD_Y+i*gy,
                         self.width-self.PAD_X, self.PAD_Y+i*gy])

    def _labels(self):
        # remove old axis labels
        for w in list(self.children):
            if getattr(w, "_axis", False):
                self.remove_widget(w)
        # X-axis: dynamic based on max_x
        if self.max_x <=  60: step = 10
        elif self.max_x <= 600: step = 100
        else:                   step = 300
        n = int(self.max_x // step) + 1
        for i in range(n):
            hz = i*step
            x = self.PAD_X + i*(self.width-2*self.PAD_X)/(n-1) - 20
            lbl = Label(text=f"{hz:d} Hz", size_hint=(None,None), size=(60,20),
                        pos=(x, self.PAD_Y-28))
            lbl._axis = True
            self.add_widget(lbl)
        # Y-axis: scientific notation
        for i in range(11):
            mag = self.max_y * i / 10
            y   = self.PAD_Y + i*(self.height-2*self.PAD_Y)/10 - 8
            for x in (self.PAD_X-68, self.width-self.PAD_X+10):
                lbl = Label(text=f"{mag:.1e}", size_hint=(None,None), size=(60,20),
                            pos=(x, y))
                lbl._axis = True
                self.add_widget(lbl)

    def redraw(self, *_):
        self.canvas.clear()
        # remove old peak labels
        for w in list(self.children):
            if getattr(w, "_peak", False):
                self.remove_widget(w)
        if not self.datasets:
            return
        peaks = []
        with self.canvas:
            self._grid()
            self._labels()
            # plot each series + record its peak
            for idx, pts in enumerate(self.datasets):
                if not pts: continue
                Color(*self.COLORS[idx % len(self.COLORS)])
                scaled = self._scale(pts)
                Line(points=scaled, width=self.LINE_W)
                fx, fy = max(pts, key=lambda p: p[1])
                sx, sy = self._scale([(fx,fy)])[0:2]
                peaks.append((fx, fy, sx, sy))
            # plot diff line if exists
            if self.diff:
                Color(*self.DIFF_CLR)
                Line(points=self._scale(self.diff), width=self.LINE_W)
        # annotate peaks
        for fx, fy, sx, sy in peaks:
            lbl = Label(text=f"▲ {fx:.1f} Hz", size_hint=(None,None), size=(85,22),
                        pos=(sx-28, sy+6))
            lbl._peak = True
            self.add_widget(lbl)
        # Δ annotation if two series
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

# ── メインアプリ ---------------------------------------------------
class FFTApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # realtime accel state
        self.rt_on  = False
        self.rt_buf = {'x':deque(maxlen=256),
                       'y':deque(maxlen=256),
                       'z':deque(maxlen=256)}
        # mic state
        self.mic_on     = False
        self.mic_buf    = deque(maxlen=4096)
        self.mic_stream = None

    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except: pass

    def _ask_perm(self, *_):
        # at startup request storage & audio permissions
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            self.btn_rt.disabled  = False
            self.btn_mic.disabled = False
            return
        need = [
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
            getattr(Permission, "MANAGE_EXTERNAL_STORAGE", None),
            getattr(Permission, "RECORD_AUDIO", None)
        ]
        if ANDROID_API >= 33:
            need += [
                Permission.READ_MEDIA_IMAGES,
                Permission.READ_MEDIA_AUDIO,
                Permission.READ_MEDIA_VIDEO
            ]
        need = [p for p in need if p]
        def _cb(perms, grants):
            ok = any(grants)
            self.btn_sel.disabled = not ok
            self.btn_rt.disabled  = not ok
            self.btn_mic.disabled = not ok
            if not ok:
                self.log("권한 거부됨 – 파일/마이크 접근이 불가합니다")
        request_permissions(need, _cb)

    # ---- realtime accel toggle ----
    def toggle_realtime(self, *_):
        self.rt_on = not self.rt_on
        self.btn_rt.text = f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel, 0)
                threading.Thread(target=self._rt_fft_loop, daemon=True).start()
            except Exception as e:
                self.log(f"가속도계 활성화 실패: {e}")
                self.rt_on = False
                self.btn_rt.text = "Realtime FFT (OFF)"
        else:
            accelerometer.disable()

    def _poll_accel(self, dt):
        if not self.rt_on:
            return False
        try:
            ax, ay, az = accelerometer.acceleration
            if None in (ax,ay,az): return
            now = time.time()
            self.rt_buf['x'].append((now, abs(ax)))
            self.rt_buf['y'].append((now, abs(ay)))
            self.rt_buf['z'].append((now, abs(az)))
        except: pass

    def _rt_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[ax])<64 for ax in ('x','y','z')):
                continue
            datasets, ymax, xmax = [], 0.0, 0.0
            for ax in ('x','y','z'):
                ts, vals = zip(*self.rt_buf[ax])
                sig = np.array(vals, dtype=float)
                n   = len(sig)
                dt  = (ts[-1]-ts[0])/(n-1) if n>1 else 1/128.0
                sig -= sig.mean(); sig *= np.hanning(n)
                freq = np.fft.fftfreq(n, d=dt)[:n//2]
                amp  = np.abs(fft(sig))[:n//2]
                mask = freq <= 50
                freq, amp = freq[mask], amp[mask]
                smooth = np.convolve(amp, np.ones(8)/8, 'same')
                datasets.append(list(zip(freq, smooth)))
                ymax = max(ymax, smooth.max()); xmax = max(xmax, freq[-1])
            Clock.schedule_once(lambda *_:
                self.graph.update_graph(datasets, [], xmax, ymax))

    # ---- mic FFT toggle ----
    def toggle_mic(self, *_):
        self.mic_on = not self.mic_on
        self.btn_mic.text = f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self._start_mic_stream()
            except Exception as e:
                self.log(f"마이크 시작 실패: {e}")
                self.mic_on = False
                self.btn_mic.text = "Mic FFT (OFF)"
        else:
            self._stop_mic_stream()

    def _start_mic_stream(self):
        self.mic_stream = sd.InputStream(
            samplerate=44100, channels=1, dtype='float32',
            blocksize=512, callback=self._on_mic_block
        )
        self.mic_stream.start()
        threading.Thread(target=self._mic_fft_loop, daemon=True).start()

    def _stop_mic_stream(self):
        try:
            self.mic_stream.stop(); self.mic_stream.close()
        except: pass

    def _on_mic_block(self, in_data, frames, time_info, status):
        if self.mic_on:
            self.mic_buf.extend(in_data[:,0])

    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.25)
            if len(self.mic_buf) < 2048:
                continue
            sig = np.array(self.mic_buf, dtype=float)
            self.mic_buf.clear()
            sig -= sig.mean(); sig *= np.hanning(len(sig))
            n  = len(sig); dt = 1/44100.0
            freq = np.fft.fftfreq(n, d=dt)[:n//2]
            amp  = np.abs(fft(sig))[:n//2]
            mask = freq <= 1500
            freq, amp = freq[mask], amp[mask]
            smooth = np.convolve(amp, np.ones(16)/16, 'same')
            pts  = list(zip(freq, smooth))
            ymax = smooth.max()
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([pts], [], 1500, ymax))

    # ---- UI build ----
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.label   = Label(text="Pick 1 or 2 CSV files", size_hint=(1,.1))
        self.btn_sel = Button(text="Select CSV", disabled=True, size_hint=(1,.1),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",   disabled=True, size_hint=(1,.1),
                              on_press=self.run_fft)
        self.btn_rt  = Button(text="Realtime FFT (OFF)", size_hint=(1,.1),
                              on_press=self.toggle_realtime)
        self.btn_mic = Button(text="Mic FFT (OFF)",      size_hint=(1,.1),
                              on_press=self.toggle_mic)

        for w in (self.label, self.btn_sel, self.btn_run,
                  self.btn_rt, self.btn_mic):
            root.add_widget(w)
        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))
        self.graph = GraphWidget(size_hint=(1,.6))
        root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # ---- CSV chooser ----
    def open_chooser(self, *_):
        if ANDROID and ANDROID_API >= 30:
            try:
                from jnius import autoclass
                Env = autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():
                    mv = ModalView(size_hint=(.8,.35))
                    box = BoxLayout(orientation='vertical', spacing=10, padding=10)
                    box.add_widget(Label(
                        text="⚠️ CSV 접근을 위해 ‘모든 파일’ 권한 필요",
                        halign="center"))
                    box.add_widget(Button(text="설정 열기", size_hint=(1,.4),
                                          on_press=lambda *_: (
                                              mv.dismiss(),
                                              self._goto_allfiles_permission())))
                    mv.add_widget(box); mv.open()
                    return
            except: pass
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,
                                          multiple=True, mime_type="text/*")
                return
            except Exception as e:
                Logger.exception("SAF picker fail"); self.log(f"SAF 오류: {e}")
        filechooser.open_file(on_selection=self.on_choose,
                              multiple=True, filters=[("CSV","*.csv")],
                              native=False,
                              path="/storage/emulated/0/Download")

    def _goto_allfiles_permission(self):
        from jnius import autoclass
        Intent   = autoclass("android.content.Intent")
        Settings = autoclass("android.provider.Settings")
        Uri      = autoclass("android.net.Uri")
        act      = autoclass("org.kivy.android.PythonActivity").mActivity
        uri = Uri.fromParts("package", act.getPackageName(), None)
        act.startActivity(Intent(
            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, uri))

    def on_choose(self, sel):
        if not sel: return
        paths = []
        for raw in sel[:2]:
            real = uri_to_file(raw)
            if not real:
                self.log("❌ 복사 실패"); return
            paths.append(real)
        self.paths = paths
        self.label.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # ---- CSV FFT ----
    def run_fft(self, *_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        res = []
        for p in self.paths:
            pts, xm, ym = self.csv_fft(p)
            if pts is None:
                self.log("CSV parse 오류"); return
            res.append((pts, xm, ym))
        if len(res)==1:
            pts, xm, ym = res[0]
            Clock.schedule_once(lambda *_: self.graph.update_graph([pts], [], xm, ym))
        else:
            (f1,x1,y1),(f2,x2,y2) = res
            diff = [(f1[i][0], abs(f1[i][1]-f2[i][1]))
                    for i in range(min(len(f1),len(f2)))]
            xm = max(x1,x2); ym = max(y1,y2, max(y for _,y in diff))
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([f1,f2], diff, xm, ym))
        Clock.schedule_once(lambda *_: setattr(self.btn_run, "disabled", False))

    @staticmethod
    def csv_fft(path: str):
        try:
            t,a = [],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a)<2: raise ValueError
            dt = (t[-1]-t[0])/len(a)
            f = np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v = np.abs(fft(a))[:len(a)//2]
            m = f <= 50; f,v = f[m], v[m]
            s = np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f,s)), 50, s.max()
        except Exception as e:
            Logger.error(f"FFT err {e}")
            return None,0,0

# ── エントリポイント ---------------------------------------------
if __name__=="__main__":
    FFTApp().run()
