"""
FFT CSV Viewer – 안정(권한·SAF) 버전
* SAF 복사는 androidstorage4kivy 사용 → pyjnius 코드 최소
* 저장소 권한 (READ/WRITE) 체크 + Android 13 미디어 권한 대응
* FileChooser native=True (SAF 피커) → 실패 시 native=False fallback
"""

import os, csv, itertools, threading, uuid, urllib.parse
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
from kivy.utils        import platform
from plyer             import filechooser, toast

# ─── Android / androidstorage ──────────────────────────────────────────
ANDROID = platform == "android"
if ANDROID:
    from androidstorage4kivy import SharedStorage
    from android.permissions import (
        check_permission, request_permissions, Permission)

# ─── SAF URI → 앱 캐시에 복사 (or file:// → 실경로) ──────────────────
def uri2path(uri: str) -> str | None:
    if not uri:
        return None
    if uri.startswith("file://"):
        real = urllib.parse.unquote(uri[7:])
        return real if os.path.exists(real) else None
    if ANDROID and uri.startswith("content://"):
        try:
            # to_downloads=False → 앱 전용 cache 로 복사
            return SharedStorage().copy_from_shared(uri, str(uuid.uuid4()), False)
        except Exception as e:
            Logger.error(f"SAF copy fail: {e}")
            return None
    return uri if os.path.exists(uri) else None

# ─── 매우 단순 그래프 (필요하면 축/격자 추가하세요) ────────────────────
class Graph(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.data  = []
        self.colors = itertools.cycle([(1,0,0),(0,1,0),(0,0,1)])
        self.bind(size=lambda *_: self.redraw())

    def update(self, *datasets):
        self.data = datasets
        self.redraw()

    def redraw(self):
        self.canvas.clear()
        if not self.data:
            return
        with self.canvas:
            for pts in self.data:
                Color(*next(self.colors))
                # x 축: 0~50Hz → (40 ~ 540)  /  y 축: 0~1 정규값 → (40 ~ 540)
                Line(points=[coord
                             for x, y in pts
                             for coord in (40 + x*10,
                                           40 + y*500)])

# ─── 메인 앱 ───────────────────────────────────────────────────────────
class FFTApp(App):

    # 라벨+토스트 동시 로그
    def log(self, msg: str):
        Logger.info(msg)
        self.lbl.text = msg
        if ANDROID:
            try: toast.toast(msg)
            except Exception: pass

    # 저장소 퍼미션
    def ask_perm(self, *_):
        if not ANDROID:
            self.btn_sel.disabled = False
            return
        need = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
        if all(check_permission(p) for p in need):
            self.btn_sel.disabled = False
        else:
            request_permissions(need,
                lambda *_: setattr(self.btn_sel, "disabled", False))

    # UI
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)

        self.lbl = Label(text="Pick 1 or 2 CSV files", size_hint=(1,.1))
        root.add_widget(self.lbl)

        self.btn_sel = Button(text="Select CSV", disabled=True,
                              size_hint=(1,.1), on_press=self.pick)
        root.add_widget(self.btn_sel)

        self.btn_run = Button(text="RUN FFT", disabled=True,
                              size_hint=(1,.1), on_press=self.run)
        root.add_widget(self.btn_run)

        root.add_widget(Button(text="EXIT", size_hint=(1,.1), on_press=self.stop))

        self.graph = Graph(size_hint=(1,.6)); root.add_widget(self.graph)

        Clock.schedule_once(self.ask_perm, 0)
        return root

    # 파일 선택
    def pick(self,*_):
        try:
            filechooser.open_file(self.on_pick, multiple=True,
                                  filters=[("CSV","*.csv")], native=True)
        except Exception:
            Logger.exception("native chooser error")
            filechooser.open_file(self.on_pick, multiple=True,
                                  filters=[("CSV","*.csv")], native=False)

    def on_pick(self, sel):
        Logger.info(f"PICK → {sel}")
        if not sel:
            return
        paths = [p for p in (uri2path(u) for u in sel[:2]) if p]
        if not paths:
            self.log("❌ copy fail"); return
        self.paths = paths
        self.lbl.text = " · ".join(os.path.basename(p) for p in paths)
        self.btn_run.disabled = False

    # FFT 실행
    def run(self,*_):
        self.btn_run.disabled = True
        threading.Thread(target=self._fft_bg, daemon=True).start()

    def _fft_bg(self):
        series=[]
        for fp in self.paths:
            t,a=[],[]
            with open(fp) as f:
                for r in csv.reader(f):
                    try: t.append(float(r[0])); a.append(float(r[1]))
                    except: pass
            if len(a)<2: continue
            dt=(t[-1]-t[0])/len(a)
            f=np.fft.fftfreq(len(a),d=dt)[:len(a)//2]
            v=np.abs(fft(a))[:len(a)//2]
            m=f<=50; f,v=f[m],v[m]; v/=v.max()
            series.append(list(zip(f, v)))
        Clock.schedule_once(lambda *_: (
            self.graph.update(*series),
            toast.toast("FFT Done") if ANDROID else None,
            setattr(self.btn_run, "disabled", False)
        ))

    # … (끝) …

if __name__ == "__main__":
    FFTApp().run()
