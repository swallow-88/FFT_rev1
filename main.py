# safe_base.py  ── 절대 안 죽는 최소 버전
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.uix.widget   import Widget
from kivy.graphics import Line, Color
import itertools, numpy as np
from plyer import filechooser

class Graph(Widget):
    def __init__(s,**kw):
        super().__init__(**kw); s.data=[]
        s.cols=itertools.cycle([(1,0,0),(0,1,0),(0,0,1)])
        s.bind(size=lambda *_: s.redraw())
    def update(s,pts):
        s.data=[pts]; s.redraw()
    def redraw(s):
        s.canvas.clear()
        if not s.data: return
        pad=40; w=max(s.width-2*pad,1); h=max(s.height-2*pad,1)
        xmax=max(x for x,_ in s.data[0]); ymax=max(y for _,y in s.data[0])
        with s.canvas:
            Color(0,1,0)
            Line(points=[pad+i/xmax*w for i,_ in s.data[0]
                         for _ in (0,)] +      # dummy
                 [pad+j/ymax*h for _,j in s.data[0]])

class Demo(App):
    def build(self):
        root=BoxLayout(orientation="vertical")
        self.lbl=Label(text="Hello"); root.add_widget(self.lbl)
        root.add_widget(Button(text="gen",on_press=self.gen))
        self.g=Graph(); root.add_widget(self.g)
        return root

    def pick(self,*_):
        filechooser.open_file(self.on_choose,multiple=True,native=False,
                              filters=[("CSV","*.csv")])
    def on_choose(self,sel):
        self.lbl.text=str(sel)

    
    def gen(self,*_):
        x=np.linspace(0,10,512); y=np.sin(x)
        self.g.update(list(zip(x,y)))




if __name__=="__main__":
    Demo().run()
