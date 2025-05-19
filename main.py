"""
FFT CSV Viewer – SAF & ‘모든-파일’ 권한 대응 안정판
"""

# ── 표준‧3rd-party --------------------------------------------------------
import os, csv, sys, traceback, threading, itertools, datetime, uuid, urllib.parse
import numpy as np
from numpy.fft import fft

from kivy.app        import App
from kivy.clock      import Clock
from kivy.logger     import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button    import Button
from kivy.uix.label     import Label
from kivy.uix.widget    import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.popup     import Popup
from kivy.graphics      import Line, Color
from kivy.utils         import platform
from plyer              import filechooser                        # (SAF fail fallback)

# ── Android 전용 모듈(있을 때만) -----------------------------------------
ANDROID = platform == "android"

toast           = None
SharedStorage   = None
Permission      = None
check_permission = request_permissions = None
ANDROID_API     = 0

if ANDROID:
    try: from plyer import toast
    except Exception: toast = None

    try: from androidstorage4kivy import SharedStorage
    except Exception: SharedStorage = None

    try:
        from android.permissions import (
            check_permission, request_permissions, Permission)
    except Exception:
        # p4a recipe가 없을 때용 더미
        check_permission  = lambda *a, **k: True
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
# -------------------------------------------------------------------------

# ── 전역 예외 → /sdcard/fft_crash.log ------------------------------------
def _dump_crash(txt: str):
    try:
        with open("/sdcard/fft_crash.log", "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n" +
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

# ── SAF URI → 앱 캐시 ------------------------------------------------------
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

# ── 간단 그래프 -----------------------------------------------------------
class GraphWidget(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.diff = [], []
        self.colors = itertools.cycle([(1,0,0), (0,1,0), (0,0,1)])
        self.pad_x = 80; self.pad_y = 30
        self.max_x = self.max_y = 1
        self.bind(size=self.redraw)

    def update_graph(self, ds, df, xm, ym):
        self.datasets, self.diff, self.max_x, self.max_y = ds, df, xm, ym
        self.redraw()

    # ---------- 도우미 ---------- #
    def _scale(self, pts):
        w, h = self.width-2*self.pad_x, self.height-2*self.pad_y
        return [c for x, y in pts
                  for c in (self.pad_x + x/self.max_x*w,
                            self.pad_y + y/self.max_y*h)]

    def _grid(self):
        gx, gy = (self.width-2*self.pad_x)/10, (self.height-2*self.pad_y)/10
        Color(.6,.6,.6)
        for i in range(11):
            Line(points=[self.pad_x+i*gx, self.pad_y,
                         self.pad_x+i*gx, self.height-self.pad_y])
            Line(points=[self.pad_x, self.pad_y+i*gy,
                         self.width-self.pad_x, self.pad_y+i*gy])

    def _labels(self):
        for w in list(self.children):
            if isinstance(w, Label): self.remove_widget(w)
        for i in range(11):
            freq = self.max_x/10*i
            x = self.pad_x+i*(self.width-2*self.pad_x)/10-20
            y = self.pad_y-30
            self.add_widget(Label(text=f"{freq:.1f} Hz",
                                  size_hint=(None,None), size=(60,20), pos=(x,y)))
        for i in range(11):
            mag = self.max_y/10*i
            y = self.pad_y+i*(self.height-2*self.pad_y)/10-10
            for x in (self.pad_x-70, self.width-self.pad_x+20):
                self.add_widget(Label(text=f"{mag:.1e}",
                                      size_hint=(None,None), size=(60,20), pos=(x,y)))

    def redraw(self,*_):
        self.canvas.clear()
        if not self.datasets:
            return
        with self.canvas:
            self._grid(); self._labels()
            col = self.colors
            for pts in self.datasets:
                Color(*next(col)); Line(points=self._scale(pts))
            if self.diff:
                Color(1,1,1); Line(points=self._scale(self.diff))

# ── 메인 앱 ---------------------------------------------------------------
class FFTApp(App):

    # --- 작은 로그 출력 --------------------------------------------------
    def log(self, msg: str):
        Logger.info(msg)
        self.label.text = msg
        if toast:
            try: toast.toast(msg)
            except Exception: pass

    # --- 저장소 권한 -----------------------------------------------------
    def _ask_perm(self,*_):

        if not ANDROID or SharedStorage:
            self.btn_sel.disabled = False
            return

        
        need = [Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE]

        # MANAGE_EXTERNAL_STORAGE 상수는 기기에 따라 없음
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

    # --- UI -------------------------------------------------------------
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.label   = Label(text="Pick 1–2 CSV files",  size_hint=(1,.1))
        self.btn_sel = Button(text="Select CSV", disabled=True, size_hint=(1,.1),
                              on_press=self.open_chooser)
        self.btn_run = Button(text="FFT RUN",   disabled=True, size_hint=(1,.1),
                              on_press=self.run_fft)

        root.add_widget(self.label)
        root.add_widget(self.btn_sel)
        root.add_widget(self.btn_run)
        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))
        self.graph = GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph)

        Clock.schedule_once(self._ask_perm, 0)
        return root

    # --- 파일 선택 -------------------------------------------------------
    def open_chooser(self,*_):
        # Android 11+ ‘모든-파일’ 권한 안내 ------------------------------
        if ANDROID and ANDROID_API >= 30:
            try:
                from jnius import autoclass
                Env = autoclass("android.os.Environment")
                if not Env.isExternalStorageManager():
                    mv = ModalView(size_hint=(.8,.35))
                    box = BoxLayout(orientation='vertical', spacing=10, padding=10)
                    box.add_widget(Label(
                        text="⚠️  CSV 파일에 접근하려면\n'모든 파일' 권한이 필요합니다.",
                        halign="center"))
                    box.add_widget(Button(text="권한 설정으로 이동",
                                          size_hint=(1,.4),
                                          on_press=lambda *_: (
                                              mv.dismiss(),
                                              self._goto_allfiles_permission())))
                    mv.add_widget(box); mv.open()
                    return
            except Exception:
                Logger.exception("ALL-FILES check 오류 (무시)")

        # ① SAF picker ---------------------------------------------------
        if ANDROID and SharedStorage:
            try:
                SharedStorage().open_file(
                    callback=self.on_choose,
                    multiple=True,
                    mime_type="text/*")   # csv 포함
                return
            except Exception as e:
                Logger.exception("SAF picker fail")
                self.log(f"SAF 선택기 오류: {e}")

        # ② 경로-chooser -------------------------------------------------
        try:
            filechooser.open_file(
                self.on_choose,
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

    # --- 선택 결과 처리 --------------------------------------------------
    def on_choose(self, sel):
        Logger.info(f"선택 결과: {sel}")
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

    # --- FFT ------------------------------------------------------------
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

    @staticmethod
    def csv_fft(path):
        try:
            t,a=[],[]
            with open(path) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except Exception: pass
            if len(a)<2: raise ValueError
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a), d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]
            s=np.convolve(v, np.ones(10)/10, 'same')
            return list(zip(f,s)), f.max(), s.max()
        except Exception as e:
            Logger.error(f"FFT err {e}")
            return None,0,0

# ------------------------------------------------------------------------
if __name__ == "__main__":
    FFTApp().run()
