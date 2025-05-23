"""
FFT CSV / Accel / Mic Viewer  –  Android SAF & Permission ready
(2025-05 안정판)
"""

# ─────────────── 기본·3rd-party ───────────────
import os, csv, sys, traceback, threading, datetime, uuid, urllib.parse, time
from collections import deque
import itertools, numpy as np
from numpy.fft import fft

from kivy.app      import App
from kivy.clock    import Clock
from kivy.logger   import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label     import Label
from kivy.uix.button    import Button
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color
from kivy.utils         import platform
from plyer              import filechooser, accelerometer

# 데스크톱 테스트용 optional
try:
    import sounddevice as sd      # 윈/맥/리눅스에서만 사용
except Exception:
    sd = None

# ─────────────── Android 특수 모듈 ─────────────
ANDROID   = platform == "android"
toast     = None
SharedStorage = None
Permission = check_permission = request_permissions = None
ANDROID_API = 0

if ANDROID:
    from jnius import autoclass, cast

    try:
        from plyer import toast
    except: pass

    try:
        from androidstorage4kivy import SharedStorage
    except: pass

    try:
        from android.permissions import (check_permission,
                                          request_permissions,
                                          Permission)
    except Exception:
        # recipe 미포함 빌드 대비
        class _Dummy: READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
        Permission = _Dummy
        check_permission  = lambda *a, **k: True
        request_permissions = lambda *a, **k: None

    try:
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except: pass

# ─────────────── 공통 유틸 ─────────────────────
def _dump(txt:str):
    """치명적 예외를 /sdcard/fft_crash.log 에 저장"""
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as fp:
            fp.write("\n"+"="*60+"\n"+datetime.datetime.now().isoformat()+"\n")
            fp.write(txt+"\n")
    except: pass
    Logger.error(txt)

def _ex(et,ev,tb):
    _dump("".join(traceback.format_exception(et,ev,tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:
            Popup(title="Python Crash",
                  content=Label(text=str(ev)), size_hint=(.9,.9)).open())
sys.excepthook = _ex


# ─────────────── SAF helper ───────────────────
def uri_to_file(u:str)->str|None:
    """SAF uri → 실제 파일(캐시 복사) path"""
    if not u: return None
    if u.startswith("file://"):
        p = urllib.parse.unquote(u[7:]);  return p if os.path.exists(p) else None
    if not u.startswith("content://"):   # 경로 문자열
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(
                u, uuid.uuid4().hex, to_downloads=False)
        except Exception as e:
            Logger.error(f"SharedStorage copy fail: {e}")
    return None


# ─────────────── 그래프 위젯 ───────────────────
class GraphWidget(Widget):
    PAD_X,PAD_Y = 80,30
    COLORS  = [(1,0,0),(0,1,0),(0,0,1)]   # 빨/초/파
    LINE_W  = 2.5
    DIFF_CLR= (1,1,1)

    def __init__(self,**kw):
        super().__init__(**kw)
        self.datasets=[]; self.diff=[]
        self.max_x=self.max_y=1
        self.bind(size=self.redraw)

    def update_graph(self,ds,df,xm,ym):
        self.datasets = [p for p in (ds or []) if p]
        self.diff     = df or []
        self.max_x    = max(1e-6,float(xm))
        self.max_y    = max(1e-6,float(ym))
        self.redraw()

    # ---------- 내부 그리기 ----------
    def _scale(self,pts):
        w,h=self.width-2*self.PAD_X, self.height-2*self.PAD_Y
        return [c for x,y in pts
                  for c in (self.PAD_X+x/self.max_x*w,
                            self.PAD_Y+y/self.max_y*h)]

    def _grid(self):
        gx,gy=(self.width-2*self.PAD_X)/10,(self.height-2*self.PAD_Y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.PAD_X+i*gx,self.PAD_Y,
                         self.PAD_X+i*gx,self.height-self.PAD_Y])
            Line(points=[self.PAD_X,self.PAD_Y+i*gy,
                         self.width-self.PAD_X,self.PAD_Y+i*gy])

    def _labels(self):
        for w in list(self.children):
            if getattr(w,"_axis",False): self.remove_widget(w)

        # X 축 : max_x 범위 따라 간격 결정
        if   self.max_x<=60:  step=10
        elif self.max_x<=600: step=100
        else:                 step=300
        n=int(self.max_x//step)+1
        for i in range(n):
            x=self.PAD_X+i*(self.width-2*self.PAD_X)/(n-1)-20
            lbl=Label(text=f"{i*step} Hz",size_hint=(None,None),
                      size=(60,20),pos=(x,self.PAD_Y-28)); lbl._axis=True
            self.add_widget(lbl)

        # Y 축
        for i in range(11):
            mag=self.max_y*i/10
            y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for x in (self.PAD_X-68,self.width-self.PAD_X+10):
                lbl=Label(text=f"{mag:.1e}",size_hint=(None,None),
                          size=(60,20),pos=(x,y)); lbl._axis=True
                self.add_widget(lbl)

    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w,"_peak",False): self.remove_widget(w)
        if not self.datasets: return

        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            for idx,pts in enumerate(self.datasets):
                if not pts: continue
                Color(*self.COLORS[idx%len(self.COLORS)])
                Line(points=self._scale(pts),width=self.LINE_W)
                fx,fy=max(pts,key=lambda p:p[1]); sx,sy=self._scale([(fx,fy)])[0:2]
                peaks.append((fx,sx,sy))
            if self.diff:
                Color(*self.DIFF_CLR); Line(points=self._scale(self.diff),width=self.LINE_W)

        for fx,sx,sy in peaks:
            lbl=Label(text=f"▲ {fx:.1f} Hz",size_hint=(None,None),
                      size=(80,22),pos=(sx-30,sy+6)); lbl._peak=True
            self.add_widget(lbl)


# ─────────────── Mic(안드) 래퍼 ───────────────
if ANDROID:
    class AndroidMic:
        """단순 16 kHz mono PCM 스트림 → deque 로 push"""
        RATE   = 16000
        CHUNK  = 1024

        def __init__(self, dq:deque):
            self.dq=dq
            self._rec=None

        def start(self):
            AudioRecord      = autoclass("android.media.AudioRecord")
            MediaRecorder    = autoclass("android.media.MediaRecorder")
            AudioFormat      = autoclass("android.media.AudioFormat")
            buf_size = AudioRecord.getMinBufferSize(
                self.RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT)
            self._rec = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                self.RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                max(buf_size, self.CHUNK*2))
            self._rec.startRecording()
            threading.Thread(target=self._loop,daemon=True).start()

        def _loop(self):
            import array, struct
            data=array.array('h',[0]*self.CHUNK)
            while self._rec and self._rec.getRecordingState()==3:
                read=self._rec.read(data,0,len(data))
                if read>0:
                    # 정규화 (-1..1)
                    self.dq.extend([s/32768.0 for s in data[:read]])

        def stop(self):
            try:
                self._rec.stop(); self._rec.release()
            except Exception: pass
            self._rec=None
else:
    AndroidMic=None


# ─────────────── 메인 앱 ──────────────────────
class FFTApp(App):
    def __init__(self,**kw):
        super().__init__(**kw)
        # accel
        self.rt_on=False
        self.rt_buf={ax:deque(maxlen=256) for ax in "xyz"}
        # mic
        self.mic_on=False
        self.mic_buf=deque(maxlen=4096)
        self._mic=None   # AndroidMic or sounddevice.stream

    # ---------- 공통 로그 ----------
    def log(self,msg):
        Logger.info(msg); self.label.text=msg
        if toast: 
            try: toast.toast(msg)
            except: pass

    # ---------- UI ----------
    def build(self):
        root=BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.label=Label(text="Pick CSV or use sensors",size_hint=(1,.1))
        self.btn_sel=Button(text="Select CSV",disabled=True,size_hint=(1,.1),
                            on_press=self.open_chooser)
        self.btn_run=Button(text="FFT RUN",disabled=True,size_hint=(1,.1),
                            on_press=self.run_fft)
        self.btn_rt =Button(text="Realtime Accel FFT (OFF)",size_hint=(1,.1),
                            on_press=self.toggle_rt)
        self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.1),
                            on_press=self.toggle_mic)
        root.add_widget(self.label); root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run); root.add_widget(self.btn_rt)
        root.add_widget(self.btn_mic)
        root.add_widget(Button(text="EXIT",size_hint=(1,.1),on_press=self.stop))
        self.graph=GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

    # ---------- 권한 ----------
    def _ask_perm(self,*_):
        if not ANDROID:
            self.btn_sel.disabled=False; return
        need=[Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE]
        aud = getattr(Permission,"RECORD_AUDIO",None)
        if aud: need.append(aud)
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled=False
        else:
            request_permissions(need, lambda *_: setattr(self.btn_sel,"disabled",False))

    # ---------- CSV ----------
    def open_chooser(self,*_):
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(callback=self.on_choose,multiple=True,mime_type="text/*"); return
            except: pass
        filechooser.open_file(on_selection=self.on_choose,multiple=True,filters=[("CSV","*.csv")])

    def on_choose(self,sel):
        if not sel: return
        self.paths=[uri_to_file(u) for u in sel[:2]]
        self.label.text=" · ".join(os.path.basename(p) for p in self.paths if p)
        self.btn_run.disabled=False

    def run_fft(self,*_):
        self.btn_run.disabled=True
        threading.Thread(target=self._fft_bg,daemon=True).start()

    def _fft_bg(self):
        out=[]; diff=[]
        for p in self.paths:
            pts,xm,ym=self._csv_fft(p)
            if pts is None: self.log("CSV parse error"); return
            out.append((pts,xm,ym))
        if len(out)==1:
            pts,xm,ym=out[0]
            Clock.schedule_once(lambda *_:self.graph.update_graph([pts],[],xm,ym))
        else:
            (p1,x1,y1),(p2,x2,y2)=out
            diff=[(p1[i][0],abs(p1[i][1]-p2[i][1])) for i in range(min(len(p1),len(p2)))]
            mx=max(x1,x2); my=max(y1,y2,max(v for _,v in diff))
            Clock.schedule_once(lambda *_:self.graph.update_graph([p1,p2],diff,mx,my))
        Clock.schedule_once(lambda *_: setattr(self.btn_run,"disabled",False))

    @staticmethod
    def _csv_fft(path):
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
            v=np.convolve(v,np.ones(10)/10,'same')
            return list(zip(f,v)),50,v.max()
        except Exception as e:
            Logger.error(f"csv_fft err {e}"); return None,0,0

    # ---------- Realtime Accel ----------
    def toggle_rt(self,*_):
        self.rt_on=not self.rt_on
        self.btn_rt.text=f"Realtime Accel FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            try:
                accelerometer.enable()
                Clock.schedule_interval(self._poll_accel,0)
                threading.Thread(target=self._rt_loop,daemon=True).start()
            except Exception as e:
                self.log(str(e)); self.toggle_rt()
        else:
            accelerometer.disable()

    def _poll_accel(self,dt):
        if not self.rt_on: return False
        ax,ay,az=accelerometer.acceleration
        if None in (ax,ay,az): return
        now=time.time()
        for v,k in zip((ax,ay,az),"xyz"):
            self.rt_buf[k].append((now,abs(v)))

    def _rt_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.rt_buf[k])<64 for k in "xyz"): continue
            ds=[]; ymax=0
            for k in "xyz":
                ts,val=zip(*self.rt_buf[k]); sig=np.asarray(val)
                sig-=sig.mean(); sig*=np.hanning(len(sig))
                dt=(ts[-1]-ts[0])/(len(sig)-1)
                f=np.fft.fftfreq(len(sig),d=dt)[:len(sig)//2]
                a=np.abs(fft(sig))[:len(sig)//2]
                m=f<=50; f,a=f[m],a[m]
                a=np.convolve(a,np.ones(8)/8,'same')
                ds.append(list(zip(f,a))); ymax=max(ymax,a.max())
            Clock.schedule_once(lambda *_: self.graph.update_graph(ds,[],50,ymax))

    # ---------- Mic ----------
    def toggle_mic(self,*_):
        self.mic_on=not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self._start_mic()
            except Exception as e:
                self.log(f"Mic start fail: {e}"); self.toggle_mic()
        else:
            self._stop_mic()

    def _start_mic(self):
        if ANDROID:
            self._mic=AndroidMic(self.mic_buf); self._mic.start()
        else:
            if sd is None: raise RuntimeError("sounddevice not installed")
            self._mic=sd.InputStream(samplerate=44100,channels=1,blocksize=1024,
                                      callback=lambda d,f,ti,st: self.mic_buf.extend(d[:,0]))
            self._mic.start()
        threading.Thread(target=self._mic_loop,daemon=True).start()

    def _stop_mic(self):
        if not self._mic: return
        if ANDROID:
            self._mic.stop()
        else:
            try: self._mic.stop(); self._mic.close()
            except: pass
        self._mic=None

    def _mic_loop(self):
        rate=16000 if ANDROID else 44100
        while self.mic_on:
            time.sleep(0.25)
            if len(self.mic_buf)<2048: continue
            sig=np.array([self.mic_buf.popleft() for _ in range(len(self.mic_buf))])
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.fftfreq(len(sig),d=1/rate)[:len(sig)//2]
            a=np.abs(fft(sig))[:len(sig)//2]
            m=f<=1500; f,a=f[m],a[m]
            a=np.convolve(a,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f,a))],[],1500,a.max()))

# ─────────────── 런 ───────────────────────────
if __name__=="__main__":
    FFTApp().run()
