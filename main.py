import os, sys, csv, itertools, traceback, threading
import numpy as np
from numpy.fft import fft

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Line, Color
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from plyer import filechooser
from android.permissions import (
    request_permissions, Permission, check_permission
)

# ───────── 플랫폼 / SAF 유틸 준비 ────────────────────── ### ADD ###
from kivy.utils import platform
from jnius import autoclass

android_api = 0
if platform == "android":
    android_api = autoclass("android.os.Build$VERSION").SDK_INT
# ------------------------------------------------------ ### ADD ###

# ───────── 글로벌 예외 팝업 ─────────────────────────────
def show_error(exc_type, exc, tb):
    txt = "".join(traceback.format_exception(exc_type, exc, tb))[:1500]
    Logger.error(txt)
    Clock.schedule_once(lambda *_:
        Popup(title="Unhandled Exception",
              content=Label(text=txt), size_hint=(.9,.9)).open())
sys.excepthook = show_error

# ───────── SAF content:// → 파일경로 변환 ───────────── ### ADD ###
def uri_to_path(uri: str) -> str:
    """
    SAF(Content-URI) 를 가능한 한 직접 경로로 변환
    실패 시 원본 URI 그대로 반환
    """
    try:
        Uri              = autoclass('android.net.Uri')
        DocumentsContract = autoclass('android.provider.DocumentsContract')
        Env              = autoclass('android.os.Environment')

        u = Uri.parse(uri)
        if uri.startswith("content://com.android.providers.downloads"):
            doc_id = DocumentsContract.getDocumentId(u)      # raw:/storage/… 
            if doc_id.startswith("raw:"):
                return doc_id.replace("raw:", "")
        elif uri.startswith("content://com.android.providers.media"):
            doc_id = DocumentsContract.getDocumentId(u)      # primary:…
            type_, rel = doc_id.split(":", 1)
            return os.path.join(Env.getExternalStorageDirectory().getAbsolutePath(), rel)
    except Exception as e:
        Logger.warning(f"URI→path 변환 실패: {e}")
    return uri
# ------------------------------------------------------ ### ADD ###


class GraphWidget(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.datasets, self.difference_dataset = [], []
        self.colors = itertools.cycle([(1,0,0), (0,1,0), (0,0,1)])
        self.diff_color, self.padding_x, self.padding_y = (1,1,1), 80, 30
        self.max_x, self.max_y = 1, 1
        self.bind(size=self._redraw)

    # ---------- 그리기 ----------
    def update_graph(self, points_list, diff_points, x_max, y_max):
        # label 제거
        for c in list(self.children):
            if isinstance(c, Label):
                self.remove_widget(c)
        self.datasets, self.difference_dataset = points_list, diff_points
        self.max_x, self.max_y = x_max, y_max
        self.canvas.clear()
        self._draw()

    def _redraw(self, *_):  # size 변경 시
        if self.datasets:
            self.canvas.clear()
            self._draw()

    def _draw(self):
        with self.canvas:
            self._draw_grid(); self._draw_axis_labels(); self._draw_right_labels()
            for pts in self.datasets:
                Color(*next(self.colors)); Line(points=[self._scale(x,y) for x,y in pts], width=1)
            if self.difference_dataset:
                Color(*self.diff_color); Line(points=[self._scale(x,y) for x,y in self.difference_dataset], width=1)

    # ---------- 헬퍼 ----------
    def _scale(self, x,y):
        sx = self.padding_x + (x/self.max_x)*(self.width-2*self.padding_x)
        sy = self.padding_y + (y/self.max_y)*(self.height-2*self.padding_y)
        return sx, sy

    def _draw_grid(self):
        gx = (self.width-2*self.padding_x)/10; gy=(self.height-2*self.padding_y)/10
        Color(.7,.7,.7)
        for i in range(11):
            Line(points=[self.padding_x+i*gx, self.padding_y,
                         self.padding_x+i*gx, self.height-self.padding_y], width=1)
            Line(points=[self.padding_x, self.padding_y+i*gy,
                         self.width-self.padding_x, self.padding_y+i*gy], width=1)

    def _draw_axis_labels(self):
        for i in range(11):
            f=(self.max_x/10)*i
            self.add_widget(Label(text=f"{f:.1f}Hz", size_hint=(None,None),
                                  size=(60,20), pos=(self.padding_x+i*(self.width-2*self.padding_x)/10-20,
                                                     self.padding_y-30)))
        for i in range(11):
            m=(self.max_y/10)*i
            self.add_widget(Label(text=f"{m:.1e}", size_hint=(None,None), size=(60,20),
                                  pos=(self.padding_x-70, self.padding_y+i*(self.height-2*self.padding_y)/10-10)))

    def _draw_right_labels(self):
        for i in range(11):
            m=(self.max_y/10)*i
            self.add_widget(Label(text=f"{m:.1e}", size_hint=(None,None), size=(60,20),
                                  pos=(self.width-self.padding_x+20,
                                       self.padding_y+i*(self.height-2*self.padding_y)/10-10)))


class FFTApp(App):
    # --------- 공용 로그 ----------
    def log(self,msg):
        Logger.info(msg); self.label.text = msg
        Clock.schedule_once(lambda *_: setattr(self.label,'text',''),3)

    # --------- UI ----------
    def build(self):
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.label  = Label(text="Select 2 CSV files", size_hint=(1,.1)); root.add_widget(self.label)
        self.select_button = Button(text="Select CSV", size_hint=(1,.1)); root.add_widget(self.select_button)
        self.run_button    = Button(text="FFT RUN", size_hint=(1,.1), disabled=True); root.add_widget(self.run_button)
        self.exit_button   = Button(text="EXIT", size_hint=(1,.1)); root.add_widget(self.exit_button)
        self.graph_widget  = GraphWidget(size_hint=(1,.6)); root.add_widget(self.graph_widget)

        self.select_button.bind(on_press=self.process_data)
        self.run_button.bind(on_press=self.on_run_fft)
        self.exit_button.bind(on_press=self.stop)

        self.ensure_permissions_and_show()
        return root

    # --------- 권한 ----------
    def ensure_permissions_and_show(self):
        perms = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if android_api >= 33:
            perms += [Permission.READ_MEDIA_IMAGES, Permission.READ_MEDIA_AUDIO, Permission.READ_MEDIA_VIDEO]
        if all(check_permission(p) for p in perms):
            self.log("권한 OK – 파일을 선택하세요."); return
        request_permissions(perms, self.on_permission_result)

    def on_permission_result(self, permissions, grants):
        if all(grants):
            self.label.text="권한 승인 완료, 파일을 선택하세요."
        else:
            self.label.text="⚠️ 저장소 권한 거부됨."

    # --------- 파일 선택 ----------
    def process_data(self, _):
        self.log("loading FILE Select")
        filechooser.open_file(on_selection=self.file_selection_callback,
                              multiple=True, filters=[("CSV files","*.csv")])

    def file_selection_callback(self, selection):
        self.log(f"file result: {selection}")
        # ★ SAF URI 처리 -------------------------------------- ### ADD ###
        if selection and selection[0].startswith("content://"):
            selection = [uri_to_path(selection[0])]
            self.log(f"converted → {selection}")
        # ------------------------------------------------------ ### ADD ###

        if not selection:
            self.label.text="CSV 파일을 선택하지 않았습니다."; self.run_button.disabled=True; self.first_file=None; return
        if len(selection)>=2:
            self.selected_files=selection[:2]
            n1,n2=(os.path.basename(p) for p in self.selected_files)
            self.label.text=f"선택 완료: {n1} & {n2}"; self.run_button.disabled=False; self.first_file=None; return
        if not getattr(self,"first_file",None):
            self.first_file=selection[0]
            self.label.text=f"1번째 파일 선택됨: {os.path.basename(self.first_file)}\n2번째 CSV를 선택하세요."
            filechooser.open_file(on_selection=self.file_selection_callback, multiple=False,
                                  filters=[("CSV files","*.csv")]); return
        self.selected_files=[self.first_file, selection[0]]
        n1,n2=(os.path.basename(p) for p in self.selected_files)
        self.label.text=f"선택 완료: {n1} & {n2}"; self.run_button.disabled=False; self.first_file=None

    # --------- FFT 실행 ----------
    def on_run_fft(self,_):
        self.log("Start FFT"); self.run_button.disabled=True
        threading.Thread(target=self.compute_and_plot, args=(self.selected_files,), daemon=True).start()

    def compute_and_plot(self, files):
        self.log("CSV to FFT CHANGE")
        results=[]
        for fp in files:
            pts,xmax,ymax=self.process_csv_and_compute_fft(fp)
            if pts is None:
                self.log("CSV PROCESS FAIL"); return
            results.append((pts,xmax,ymax))
        if len(results)==1:
            f1,x1,y1=results[0]
            Clock.schedule_once(lambda dt: self.graph_widget.update_graph([f1],[],x1,y1)); return
        (f1,x1,y1),(f2,x2,y2)=results
        diff=[(f1[i][0], abs(f1[i][1]-f2[i][1])) for i in range(min(len(f1),len(f2)))]
        x_max=max(x1,x2); y_max=max(y1,y2,max(y for _,y in diff))
        Clock.schedule_once(lambda dt: self.graph_widget.update_graph([f1,f2],diff,x_max,y_max))
        Clock.schedule_once(lambda dt: setattr(self.run_button,'disabled',False))

    # --------- FFT util ----------
    def process_csv_and_compute_fft(self, fp):
        try:
            t,acc=[],[]
            with open(fp,'r') as f:
                for r in csv.reader(f): 
                    try: t.append(float(r[0])); acc.append(float(r[1]))
                    except: continue
            if len(acc)<2: raise ValueError("데이터 부족")
            dt=(t[-1]-t[0])/len(acc)
            freq=np.fft.fftfreq(len(acc),d=dt)[:len(acc)//2]
            vals=np.abs(fft(acc))[:len(acc)//2]
            mask=freq<=50; freq=freq[mask]; vals=vals[mask]
            smooth=np.convolve(vals, np.ones(10)/10, mode='same')
            return list(zip(freq,smooth)), max(freq), max(smooth)
        except Exception as e:
            Logger.error(f"FFT: {e}"); return None,0,0


if __name__=="__main__":
    FFTApp().run()
