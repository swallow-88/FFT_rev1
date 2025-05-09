##############################################################################
# 0)  imports
##############################################################################
import os, csv, sys, traceback, threading, itertools
import numpy as np
from numpy.fft import fft

from kivy.app    import App
from kivy.clock  import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.uix.widget   import Widget
from kivy.graphics     import Line, Color
from plyer             import filechooser         # ← Kivy FileChooser
##############################################################################
# 1)  예외를 파일에 기록 (PC 없이도 확인)
##############################################################################
def dump_crash(et, ev, tb):
    txt = "".join(traceback.format_exception(et, ev, tb))
    try:
        with open("/sdcard/fft_crash.log","a",encoding="utf-8") as fp:
            fp.write(txt+"\n"+"="*60+"\n")
    except Exception:
        pass
    Logger.error(txt)
sys.excepthook = dump_crash
##############################################################################
# 2)  아주 단순한 그래프 위젯 (변경 없음)
##############################################################################
class GraphWidget(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.data = []; self.colors = itertools.cycle([(1,0,0),(0,1,0),(0,0,1)])
        self.bind(size=lambda *_: self.redraw())

    def update(self, *datasets):
        self.data = datasets; self.redraw()

    def redraw(self):
        self.canvas.clear()
        if not self.data: return
        with self.canvas:
            for pts in self.data:
                Color(*next(self.colors))
                Line(points=[c for x,y in pts for c in (x*10+40, y*100+40)])

##############################################################################
# 3)  메인 앱 - 파일 선택 → FFT
##############################################################################
class FFTApp(App):
    def build(self):
        root = BoxLayout(orientation="vertical",padding=10,spacing=10)
        self.info = Label(text="CSV 두 개를 고르세요"); root.add_widget(self.info)
        root.add_widget(Button(text="Select CSV",on_press=self.pick))
        self.run = Button(text="FFT RUN",disabled=True,on_press=self.do_fft)
        root.add_widget(self.run)
        self.graph = GraphWidget(); root.add_widget(self.graph)
        return root

    def pick(self,*_):
        filechooser.open_file(
            on_selection=self.got,
            multiple=True,
            filters=[("CSV","*.csv")],
            native=False)        #  ← Kivy FileChooser (실경로 반환)

    def got(self, sel):
        Logger.info(f"Chooser ⇒ {sel}")
        if not sel: return
        self.paths = sel[:2]          # 1 개 또는 2 개
        self.info.text = " · ".join(os.path.basename(p) for p in self.paths)
        self.run.disabled = False

    def do_fft(self,*_):
        self.run.disabled = True
        threading.Thread(target=self._fft,daemon=True).start()

    def _fft(self):
        out=[]
        for p in self.paths:
            try:
                t,a=[],[]
                with open(p) as f:
                    for r in csv.reader(f):
                        t.append(float(r[0])); a.append(float(r[1]))
                dt=(t[-1]-t[0])/len(a)
                f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
                v=np.abs(fft(a))[:len(a)//2]
                out.append(list(zip(f/5, v/max(v))))    # 그냥 축소해서 표시
            except Exception as e:
                Logger.error(f"FFT err {e}")
                return
        Clock.schedule_once(lambda *_: (self.graph.update(*out),
                                        setattr(self.run,"disabled",False)))

##############################################################################
if __name__ == "__main__":
    FFTApp().run()
