# ============================================================
#  FFT Viewer  – CSV / Realtime Accelerometer / Mic (AudioRecord)
# ============================================================

import os, csv, sys, threading, itertools, datetime, traceback, time, uuid, \
       urllib.parse
from collections import deque

import numpy as np
from numpy.fft import fft

from kivy.app           import App
from kivy.clock         import Clock
from kivy.logger        import Logger
from kivy.utils         import platform
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color

from plyer               import filechooser, accelerometer, toast

# ---------- Android 전용 ----------
ANDROID = platform == "android"
SharedStorage = None
Permission = check_permission = request_permissions = None
ANDROID_API = 0

if ANDROID:
    try:
        from androidstorage4kivy import SharedStorage
    except Exception:
        SharedStorage = None
    try:
        from android.permissions import (
            Permission, check_permission, request_permissions)
    except Exception:
        class _P: READ_EXTERNAL_STORAGE=WRITE_EXTERNAL_STORAGE=\
                  READ_MEDIA_IMAGES=READ_MEDIA_AUDIO=READ_MEDIA_VIDEO=\
                  MANAGE_EXTERNAL_STORAGE=RECORD_AUDIO=""
        Permission=_P;check_permission=lambda *_:True;request_permissions=lambda *_:None
    try:
        from jnius import autoclass, cast
        ANDROID_API = autoclass("android.os.Build$VERSION").SDK_INT
    except Exception:
        ANDROID_API = 0

# ---------- 공통 유틸 ----------
def _dump_crash(msg:str):
    try:
        with open("/sdcard/fft_crash.log","a",encoding="utf-8") as fp:
            fp.write("\n"+"="*60+"\n"+datetime.datetime.now().isoformat()+"\n")
            fp.write(msg+"\n")
    except Exception: pass
    Logger.error(msg)

def _ex(et, ev, tb):
    _dump_crash("".join(traceback.format_exception(et,ev,tb)))
    if ANDROID:
        Clock.schedule_once(lambda *_:Popup(title="Python Crash",
            content=Label(text=str(ev)),size_hint=(.9,.9)).open())
sys.excepthook=_ex

def uri_to_file(u:str)->str|None:
    if not u: return None
    if u.startswith("file://"):
        real=urllib.parse.unquote(u[7:]); return real if os.path.exists(real) else None
    if not u.startswith("content://"):
        return u if os.path.exists(u) else None
    if ANDROID and SharedStorage:
        try:
            return SharedStorage().copy_from_shared(u, uuid.uuid4().hex, False)
        except Exception as e: Logger.error(f"SAF copy fail {e}")
    return None

# ---------- 그래프 위젯 ----------
class GraphWidget(Widget):
    PAD_X, PAD_Y = 80, 30
    COLORS = [(1,0,0),(0,1,0),(0,0,1)]
    LINE_W = 2.4
    def __init__(self,**kw):
        super().__init__(**kw)
        self.datasets=[]; self.diff=[]
        self.max_x=self.max_y=1
        self.bind(size=self.redraw)

    def update_graph(self,ds,df,xm,ym):
        self.datasets=[d for d in ds if d]
        self.diff=df or []
        self.max_x=max(1e-6,float(xm)); self.max_y=max(1e-6,float(ym))
        self.redraw()

    # ---------- 내부 ----------
    def _scale(self,pts):
        w,h=self.width-2*self.PAD_X,self.height-2*self.PAD_Y
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
        # 기존 축 라벨 삭제
        for w in list(self.children):
            if getattr(w,"_axis",False): self.remove_widget(w)
        # X축 간격 선택
        step=10 if self.max_x<=60 else 100 if self.max_x<=600 else 300
        n=int(self.max_x//step)+1
        for i in range(n):
            x=self.PAD_X+i*(self.width-2*self.PAD_X)/(n-1)-20
            l=Label(text=f"{i*step:d} Hz",size_hint=(None,None),
                    size=(60,20),pos=(x,self.PAD_Y-28)); l._axis=True
            self.add_widget(l)
        # Y축
        for i in range(11):
            mag=self.max_y*i/10; y=self.PAD_Y+i*(self.height-2*self.PAD_Y)/10-8
            for x in (self.PAD_X-68,self.width-self.PAD_X+10):
                l=Label(text=f"{mag:.1e}",size_hint=(None,None),
                        size=(60,20),pos=(x,y)); l._axis=True; self.add_widget(l)

    def redraw(self,*_):
        self.canvas.clear()
        for w in list(self.children):
            if getattr(w,"_peak",False): self.remove_widget(w)
        if not self.datasets: return
        peaks=[]
        with self.canvas:
            self._grid(); self._labels()
            for idx,pts in enumerate(self.datasets):
                Color(*self.COLORS[idx%len(self.COLORS)])
                Line(points=self._scale(pts),width=self.LINE_W)
                fx,fy=max(pts,key=lambda p:p[1]); sx,sy=self._scale([(fx,fy)])[:2]
                peaks.append((fx,fy,sx,sy))
            if self.diff:
                Color(1,1,1); Line(points=self._scale(self.diff),width=self.LINE_W)
        for fx,fy,sx,sy in peaks:
            l=Label(text=f"▲{fx:.1f}Hz",size_hint=(None,None),
                    size=(80,20),pos=(sx-25,sy+5)); l._peak=True; self.add_widget(l)

# ---------- Mic (AudioRecord) ----------
if ANDROID:
    AUDIO_SOURCE = 1  # MediaRecorder.AudioSource.MIC
    AUDIO_FMT     = 2 # AudioFormat.ENCODING_PCM_16BIT
    CH_IN_MONO    = 16 # AudioFormat.CHANNEL_IN_MONO
    AudioRecord   = autoclass("android.media.AudioRecord")
    AudioFormat   = autoclass("android.media.AudioFormat")
    AudioTrackLen = autoclass("android.media.AudioRecord").getMinBufferSize

def start_audio_record(buf, samplerate=44100, block=2048):
    """별도 스레드에서 AudioRecord 로 PCM을 buf(deque)에 push"""
    if not ANDROID: raise RuntimeError("AudioRecord only on Android")

    minsz = AudioTrackLen(samplerate,
              AudioFormat.CHANNEL_IN_MONO,
              AudioFormat.ENCODING_PCM_16BIT)
    rec = AudioRecord(AUDIO_SOURCE,samplerate,
                      AudioFormat.CHANNEL_IN_MONO,
                      AudioFormat.ENCODING_PCM_16BIT,
                      max(minsz,block*2))
    if rec.getState()!=AudioRecord.STATE_INITIALIZED:
        raise RuntimeError("AudioRecord unavailable")
    rec.startRecording()
    def _loop():
        arr = bytearray(block*2)
        mv  = memoryview(arr)
        while rec.getRecordingState()==AudioRecord.RECORDSTATE_RECORDING:
            n=rec.read(mv, len(arr))
            if n>0:
                pcm=np.frombuffer(arr[:n],dtype=np.int16).astype(np.float32)
                buf.extend(pcm/32768.0)
    t=threading.Thread(target=_loop,daemon=True); t.start()
    return rec

# ---------- 메인 앱 ----------
class FFTApp(App):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.paths=[]
        # accelerometer
        self.rt_on=False
        self.acc_buf={ax:deque(maxlen=256) for ax in 'xyz'}
        # microphone
        self.mic_on=False
        self.mic_buf=deque(maxlen=8192)
        self.mic_rec=None

    # ---------- UI ----------
    def build(self):
        root=BoxLayout(orientation='vertical',spacing=8,padding=8)
        self.status=Label(text="Select CSV or use realtime buttons",size_hint=(1,.08))
        root.add_widget(self.status)
        self.btn_sel=Button(text="Select CSV",size_hint=(1,.09),on_press=self.select_csv)
        self.btn_run=Button(text="FFT RUN",size_hint=(1,.09),disabled=True,on_press=self.run_fft)
        self.btn_rt =Button(text="Realtime FFT (OFF)",size_hint=(1,.09),on_press=self.toggle_rt)
        self.btn_mic=Button(text="Mic FFT (OFF)",size_hint=(1,.09),on_press=self.toggle_mic)
        root.add_widget(self.btn_sel);root.add_widget(self.btn_run)
        root.add_widget(self.btn_rt); root.add_widget(self.btn_mic)
        root.add_widget(Button(text="EXIT",size_hint=(1,.09),on_press=self.stop))
        self.graph=GraphWidget(size_hint=(1,.56)); root.add_widget(self.graph)
        Clock.schedule_once(self._ask_perm,0)
        return root

    def log(self,msg): Logger.info(msg); self.status.text=msg; toast.toast(msg) if ANDROID else None

    # ---------- permissions ----------
    def _ask_perm(self,*_):
        if not ANDROID or SharedStorage:
            self.btn_sel.disabled=False; return
        need=[Permission.READ_EXTERNAL_STORAGE,Permission.WRITE_EXTERNAL_STORAGE,
              Permission.RECORD_AUDIO]
        if ANDROID_API>=30:
            need.append(getattr(Permission,"MANAGE_EXTERNAL_STORAGE",""))
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled=False
        else:
            request_permissions(need,lambda *_:self._ask_perm())

    # ---------- CSV ----------
    def select_csv(self,*_):
        filechooser.open_file(on_selection=self._csv_chosen,multiple=True,
                              filters=[("CSV","*.csv")],native=False)
    def _csv_chosen(self,sel):
        if not sel: return
        self.paths=[uri_to_file(s) for s in sel[:2]]
        self.btn_run.disabled=False
        self.log("CSV ready")

    def csv_fft(self,path):
        t,a=[],[]
        with open(path) as f:
            for r in csv.reader(f):
                try:t.append(float(r[0]));a.append(float(r[1]))
                except:pass
        if len(a)<2: return None
        dt=(t[-1]-t[0])/len(a)
        f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
        v=np.abs(fft(a))[:len(a)//2]
        m=f<=50; f,v=f[m],v[m]
        v=np.convolve(v,np.ones(10)/10,'same')
        return list(zip(f,v)),50,v.max()

    def run_fft(self,*_):
        def _bg():
            out=[]
            for p in self.paths:
                r=self.csv_fft(p)
                if not r: self.log("CSV parse error"); return
                out.append(r)
            if len(out)==1:
                pts,xm,ym=out[0]
                Clock.schedule_once(lambda *_:self.graph.update_graph([pts],[],xm,ym))
            else:
                (f1,x1,y1),(f2,x2,y2)=out
                diff=[(f1[i][0],abs(f1[i][1]-f2[i][1]))
                      for i in range(min(len(f1),len(f2)))]
                xm=max(x1,x2); ym=max(y1,y2,max(y for _,y in diff))
                Clock.schedule_once(lambda *_:self.graph.update_graph([f1,f2],diff,xm,ym))
        threading.Thread(target=_bg,daemon=True).start()

    # ---------- accelerometer realtime ----------
    def toggle_rt(self,*_):
        self.rt_on=not self.rt_on
        self.btn_rt.text=f"Realtime FFT ({'ON' if self.rt_on else 'OFF'})"
        if self.rt_on:
            accelerometer.enable(); Clock.schedule_interval(self._acc_poll,0)
            threading.Thread(target=self._acc_fft_loop,daemon=True).start()
        else:
            accelerometer.disable()
    def _acc_poll(self,dt):
        if not self.rt_on: return False
        ax,ay,az=accelerometer.acceleration
        if None in (ax,ay,az): return
        now=time.time()
        for val,axn in zip((ax,ay,az),'xyz'):
            self.acc_buf[axn].append((now,abs(val)))
    def _acc_fft_loop(self):
        while self.rt_on:
            time.sleep(0.5)
            if any(len(self.acc_buf[a])<64 for a in 'xyz'): continue
            ds=[]; ymax=0
            for axn in 'xyz':
                ts,val=zip(*self.acc_buf[axn])
                sig=np.asarray(val,float); n=len(sig)
                dt=(ts[-1]-ts[0])/(n-1)
                sig-=sig.mean(); sig*=np.hanning(n)
                f=np.fft.fftfreq(n,d=dt)[:n//2]; amp=np.abs(fft(sig))[:n//2]
                m=f<=50; f,amp=f[m],amp[m]
                amp=np.convolve(amp,np.ones(8)/8,'same')
                ds.append(list(zip(f,amp))); ymax=max(ymax,amp.max())
            Clock.schedule_once(lambda *_:self.graph.update_graph(ds,[],50,ymax))

    # ---------- microphone ----------
    def toggle_mic(self,*_):
        self.mic_on=not self.mic_on
        self.btn_mic.text=f"Mic FFT ({'ON' if self.mic_on else 'OFF'})"
        if self.mic_on:
            try:
                self.mic_rec=start_audio_record(self.mic_buf)
                threading.Thread(target=self._mic_fft_loop,daemon=True).start()
            except Exception as e:
                self.log(f"Mic start fail: {e}"); self.mic_on=False
                self.btn_mic.text="Mic FFT (OFF)"
        else:
            try:self.mic_rec.stop();self.mic_rec.release()
            except Exception:pass
    def _mic_fft_loop(self):
        while self.mic_on:
            time.sleep(0.3)
            if len(self.mic_buf)<4096: continue
            sig=np.array([self.mic_buf.popleft() for _ in range(4096)],float)
            sig-=sig.mean(); sig*=np.hanning(len(sig))
            f=np.fft.fftfreq(len(sig),1/44100.0)[:len(sig)//2]
            amp=np.abs(fft(sig))[:len(sig)//2]
            m=f<=1500; f,amp=f[m],amp[m]
            amp=np.convolve(amp,np.ones(16)/16,'same')
            Clock.schedule_once(lambda *_:
                self.graph.update_graph([list(zip(f,amp))],[],1500,amp.max()))

# ---------- run ----------
if __name__=="__main__":
    FFTApp().run()
