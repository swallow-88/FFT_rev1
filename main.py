import os
import csv
import numpy as np
import itertools
from numpy.fft import fft
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.clock import Clock
from plyer import filechooser
from kivy.logger import Logger
from android.permissions import request_permissions, Permission, check_permission

# main.py 맨 위
import sys, traceback

from kivy.utils import platform
from jnius import autoclass

android_api = 0
if platform == "android":
    android_api = autoclass("android.os.Build$VERSION").SDK_INT


def show_error(exc_type, exc, tb):
    txt = "".join(traceback.format_exception(exc_type, exc, tb))[:1500]
    Logger.error(txt)
    Clock.schedule_once(lambda *_:
        Popup(title="Unhandled Exception", content=Label(text=txt), size_hint=(.9,.9)).open())
sys.excepthook = show_error# main.py 맨 위

import threading

class GraphWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datasets = []            # FFT 결과들
        self.difference_dataset = []  # 차이 그래프
        self.colors = itertools.cycle([(1,0,0),(0,1,0),(0,0,1)])
        self.diff_color = (1,1,1)
        self.padding_x = 80
        self.padding_y = 30
        self.max_x = 1
        self.max_y = 1

        self.bind(size=self.on_size)

    def update_graph(self, points_list, diff_points, x_max, y_max):
        # 기존 Label 위젯 제거 (레이블 재생성을 위해)
        for child in list(self.children):
            if isinstance(child, Label):
                self.remove_widget(child)

        self.datasets = points_list
        self.difference_dataset = diff_points
        self.max_x = x_max
        self.max_y = y_max

        self.canvas.clear()
        self.draw_graph()

    def on_size(self, *args):
        if self.datasets:
            self.canvas.clear()
            self.draw_graph()

    def draw_graph(self):
        with self.canvas:
            self.draw_grid()
            self.draw_axis_labels()
            self.draw_right_axis_labels()

            # FFT 결과들
            for points in self.datasets:
                Color(*next(self.colors))
                Line(points=[self.scale_point(x,y) for x,y in points], width=1)

            # 차이 그래프
            #Color(*self.diff_color)
            #Line(points=[self.scale_point(x,y) for x,y in self.difference_dataset], width=1)

            # 차이 그래프 (데이터가 있을 때만)
            if self.difference_dataset:
                Color(*self.diff_color)
                Line(points=[self.scale_point(x,y) for x,y in self.difference_dataset], width=1)
    
    def scale_point(self, x, y):
        scaled_x = self.padding_x + (x/self.max_x) * (self.width - 2*self.padding_x)
        scaled_y = self.padding_y + (y/self.max_y) * (self.height - 2*self.padding_y)
        return scaled_x, scaled_y

    def draw_grid(self):
        gx = (self.width - 2*self.padding_x) / 10
        gy = (self.height - 2*self.padding_y) / 10
        Color(0.7,0.7,0.7)
        for i in range(11):
            # 세로
            Line(points=[self.padding_x + i*gx, self.padding_y,
                          self.padding_x + i*gx, self.height-self.padding_y], width=1)
            # 가로
            Line(points=[self.padding_x, self.padding_y + i*gy,
                          self.width-self.padding_x, self.padding_y + i*gy], width=1)

    def draw_axis_labels(self):
        # X축 레이블
        for i in range(11):
            freq = (self.max_x/10)*i
            x = self.padding_x + i*(self.width-2*self.padding_x)/10 - 20
            y = self.padding_y - 30
            lbl = Label(text=f"{freq:.1f}Hz", size_hint=(None,None),
                        size=(60,20), pos=(x,y))
            self.add_widget(lbl)

        # Y축(왼쪽)
        for i in range(11):
            mag = (self.max_y/10)*i
            y = self.padding_y + i*(self.height-2*self.padding_y)/10 - 10
            x = self.padding_x - 70
            lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                        size=(60,20), pos=(x,y))
            self.add_widget(lbl)

    def draw_right_axis_labels(self):
        # Y축(오른쪽) 차이용
        for i in range(11):
            mag = (self.max_y/10)*i
            y = self.padding_y + i*(self.height-2*self.padding_y)/10 - 10
            x = self.width - self.padding_x + 20
            lbl = Label(text=f"{mag:.1e}", size_hint=(None,None),
                        size=(60,20), pos=(x,y))
            self.add_widget(lbl)


class FFTApp(App):

    def log(self, msg):
        """콘솔(Logcat) + 화면 라벨 모두 출력"""
        Logger.info(msg)
        self.label.text = msg
        Clock.schedule_once(lambda dt: setattr(self.label, "text", ""), 3)

    
    def build(self):

        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.label = Label(text="Select 2 CSV files", size_hint=(1, .1))
        self.layout.add_widget(self.label)

        self.select_button = Button(text="Select CSV", size_hint=(1, .1))
        self.select_button.bind(on_press=self.process_data)
        self.layout.add_widget(self.select_button)

        self.run_button = Button(text="FFT RUN", size_hint=(1, .1), disabled=True)
        self.run_button.bind(on_press=self.on_run_fft)
        self.layout.add_widget(self.run_button)

        self.exit_button = Button(text="EXIT", size_hint=(1, .1))
        self.exit_button.bind(on_press=self.stop)
        self.layout.add_widget(self.exit_button)

        self.graph_widget = GraphWidget(size_hint=(1, .6))
        self.layout.add_widget(self.graph_widget)

        # 2️⃣ 위젯이 준비된 뒤 권한 체크
        self.ensure_permissions_and_show()

        return self.layout

        '''
        # 런타임 권한 요청
        request_permissions([
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE
        ])


        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.label = Label(text="Select 2 CSV FILE", size_hint=(1,0.1))
        self.layout.add_widget(self.label)

        # 1) 파일 선택만 담당할 버튼
        self.select_button = Button(text="Select 2 CSV FILE", size_hint=(1,0.1))
        self.select_button.bind(on_press=self.process_data)
        self.layout.add_widget(self.select_button)

        # 2) FFT 실행 버튼 (초기에는 비활성화)
        self.run_button = Button(text="FFT RUN", size_hint=(1,0.1), disabled=True)
        self.run_button.bind(on_press=self.on_run_fft)
        self.layout.add_widget(self.run_button)
  

        self.exit_button = Button(text="EXIT", size_hint=(1,0.1))
        self.exit_button.bind(on_press=self.stop)
        self.layout.add_widget(self.exit_button)

        self.graph_widget = GraphWidget(size_hint=(1,0.6))
        self.layout.add_widget(self.graph_widget)

        self.ensure_permissions_and_show()
         
        return self.layout
        #self.first_file = None
        #return self.layout
        '''
    
# ─── imports ───────────────────────────────────────────────────────────

    def ensure_permissions_and_show(self):
        base_perms = [Permission.READ_EXTERNAL_STORAGE,
                      Permission.WRITE_EXTERNAL_STORAGE]

        extra = []
        if android_api >= 33:          # Android 13+
            extra = [Permission.READ_MEDIA_IMAGES,
                     Permission.READ_MEDIA_AUDIO,
                     Permission.READ_MEDIA_VIDEO]

        needed = base_perms + extra

        if all(check_permission(p) for p in needed):
            self.log("권한 OK – 파일을 선택하세요.")
            return

        request_permissions(needed, self.on_permission_result)

 

    def on_permission_result(self, permissions, grants):
        if all(grants):
            self.label.text = "권한 승인 완료, 파일을 선택하세요."
        else:
            self.label.text = "⚠️ 저장소 권한 거부됨. 설정에서 권한을 켜주세요."
            return
        
    
    def process_data(self, instance):
        self.log("loading FILE Select")
        filechooser.open_file(
            on_selection=self.file_selection_callback,
            multiple=True,   # 가능하면 한 번에 여러 개 고를 수 있게
            filters=[("CSV files", "*.csv")]
        )


    def file_selection_callback(self, selection):
        self.log(f"file result:{selection}")
        Logger.info(f"FileChooser: {selection}")
    
        if not selection:
            self.label.text = "CSV 파일을 선택하지 않았습니다."
            self.run_button.disabled = True
            self.first_file = None
            return
    
        # ── A) 한 번에 2 개 이상 선택한 경우 ─────────────────────────────
        if len(selection) >= 2:
            self.selected_files = selection[:2]
            n1, n2 = (os.path.basename(p) for p in self.selected_files)
            self.label.text = f"선택 완료: {n1} & {n2}"
            self.run_button.disabled = False
            self.first_file = None
            return
    
        # ── B) 한 번에 1 개씩 고르는 워크플로 ────────────────────────────
        if not hasattr(self, "first_file") or self.first_file is None:
            self.first_file = selection[0]
            self.label.text = (
                f"1번째 파일 선택됨:\n{os.path.basename(self.first_file)}\n"
                "이제 2번째 CSV를 선택하세요."
            )
            filechooser.open_file(
                on_selection=self.file_selection_callback,
                multiple=False,
                filters=[("CSV files", "*.csv")],
            )
            return
    
        # C) ‑ 두 번째 파일까지 확보
        self.selected_files = [self.first_file, selection[0]]
        n1, n2 = (os.path.basename(p) for p in self.selected_files)
        self.label.text = f"선택 완료: {n1} & {n2}"
        self.run_button.disabled = False
        self.first_file = None

        '''
        if not selection:
            return

        # 첫 번째 파일을 아직 선택하지 않았다면
        if not hasattr(self, 'first_file'):
            self.first_file = selection[0]
            self.label.text = f"1번째: {os.path.basename(self.first_file)}\n2번째 파일을 선택하세요."
            # 두 번째 파일 선택을 위해 다시 파일 선택기 열기
            filechooser.open_file(
                on_selection=self.file_selection_callback,
                multiple=False,
                filters=[("CSV files", "*.csv")]
            )
            return

        # 두 번째 파일 선택
        self.selected_files = [self.first_file, selection[0]]
        names = [os.path.basename(p) for p in self.selected_files]
        self.label.text = f"선택: {names[0]}, {names[1]}"
        self.run_button.disabled = False
        # 선택 완료했으니 first_file 속성 지워도 무방
        del self.first_file
        '''
        '''
        if not selection:
            self.label.text = "CSV 파일을 선택하세요."
            self.run_button.disabled = True
            return

        # 1개 또는 2개까지 파일을 취급
        self.selected_files = selection[:2]
        names = [os.path.basename(p) for p in self.selected_files]
        # “파일1, 파일2” 또는 “파일1” 로 표시
        self.label.text = "선택: " + ", ".join(names)
        self.run_button.disabled = False

        
        print("DEBUG selection:", selection)  # ← 추가
        '''


    '''
    def file_selection_callback(self, selection):
        if not selection:
            return
        path = selection[0]

        # 1) 첫 번째 파일 선택
        if not selection:
            return
        # 첫 번째 선택인지, 두 번째 선택인지 구분
        if not hasattr(self, 'first_file'):
            self.first_file = selection[0]
            self.label.text = f"1st: {os.path.basename(self.first_file)}\nSelect 2nd file"
        else:
            self.selected_files = [ self.first_file, selection[0] ]
            names = [os.path.basename(p) for p in self.selected_files]
            self.label.text = f"Select: {names[0]}, {names[1]}"
            self.run_button.disabled = False
    '''


       

    def on_run_fft(self, _):
        self.log("Start FFT")
        self.run_button.disabled = True
        self.label.text = "FFT 계산 중…"
        threading.Thread(target=self.compute_and_plot,
                         args=(self.selected_files,), daemon=True).start()

 
    def compute_and_plot(self, files):
        '''
        f1, x1, y1 = self.process_csv_and_compute_fft(files[0])
        f2, x2, y2 = self.process_csv_and_compute_fft(files[1])
        if f1 is None or f2 is None:
            Clock.schedule_once(lambda dt: setattr(self.label, 'text', "ERROR PROCESS CSV FILE"))
            return

        diff = [(f1[i][0], abs(f1[i][1]-f2[i][1])) for i in range(min(len(f1),len(f2)))]
        x_max = max(x1, x2)
        y_max = max(y1, y2, max(y for _,y in diff))

        # UI 스레드에서 그래프 갱신
        Clock.schedule_once(lambda dt:
            self.graph_widget.update_graph([f1,f2], diff, x_max, y_max)
        )
        '''
        '''
        results = []
        for fp in files:
            f, x_max, y_max = self.process_csv_and_compute_fft(fp)
            if f is None:
                Clock.schedule_once(lambda dt:
                    setattr(self.label, 'text', "CSV 처리 중 오류가 발생했습니다.")
                )
                return
            results.append((f, x_max, y_max))

        # 단일 파일인 경우
        if len(results) == 1:
            f1, x1, y1 = results[0]
            Clock.schedule_once(lambda dt:
                self.graph_widget.update_graph([f1], [], x1, y1)
            )
            return

        # 두 파일인 경우: 두 그래프 + 차이 그래프
        (f1, x1, y1), (f2, x2, y2) = results
        diff = [(f1[i][0], abs(f1[i][1] - f2[i][1]))
                for i in range(min(len(f1), len(f2)))]
        x_max = max(x1, x2)
        y_max = max(y1, y2, max(y for _, y in diff))
        Clock.schedule_once(lambda dt:
            self.graph_widget.update_graph([f1, f2], diff, x_max, y_max)
        )
        '''
        self.log("CSV to FFT CHANGE")
        
        results=[]
        for fp in files:
            pts,xmax,ymax = self.process_csv_and_compute_fft(fp)
            if pts is None:
                self.log("CSV PROCESS FAIL")
                Clock.schedule_once(lambda dt: setattr(self.label,'text',"CSV 오류"))
                return
            results.append((pts,xmax,ymax))

        if len(results)==1:
            f1,x1,y1 = results[0]
            Clock.schedule_once(lambda dt: self.graph_widget.update_graph([f1],[],x1,y1)); return

        (f1,x1,y1),(f2,x2,y2)=results
        diff=[(f1[i][0], abs(f1[i][1]-f2[i][1])) for i in range(min(len(f1),len(f2)))]
        x_max=max(x1,x2); y_max=max(y1,y2,max(y for _,y in diff))
        Clock.schedule_once(lambda dt: self.graph_widget.update_graph([f1,f2],diff,x_max,y_max))
        Clock.schedule_once(lambda dt: setattr(self.run_button, "disabled", False))

    
    '''
    def process_csv_and_compute_fft(self, filepath):
        try:
            time = []; acc = []
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        t = float(row[0]); a = float(row[1])
                        time.append(t); acc.append(a)
                    except:
                        continue
            n = len(acc)
            if n < 2:
                raise ValueError("ERROR DATA SIZE")
            # 시간 간격
            dt = (time[-1] - time[0]) / n
            freq = np.fft.fftfreq(n, d=dt)[:n//2]
            vals = np.abs(fft(acc))[:n//2]

            # 50Hz 이하만
            mask = freq <= 50
            freq = freq[mask]
            vals = vals[mask]

            # 간단 스무딩
            window = np.ones(10)/10
            smooth = np.convolve(vals, window, mode='same')
            points = list(zip(freq, smooth))
            return points, max(freq), max(smooth)
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {e}")
            return None, 0, 0
    '''

    def process_csv_and_compute_fft(self, fp):
        try:
            t,acc = [],[]
            with open(fp,'r') as f:
                for row in csv.reader(f):
                    try: t.append(float(row[0])); acc.append(float(row[1]))
                    except: continue
            if len(acc)<2: raise ValueError("데이터 부족")
            dt = (t[-1]-t[0])/len(acc)
            freq = np.fft.fftfreq(len(acc),d=dt)[:len(acc)//2]
            vals = np.abs(fft(acc))[:len(acc)//2]
            mask = freq<=50; freq=freq[mask]; vals=vals[mask]
            smooth = np.convolve(vals, np.ones(10)/10, mode='same')
            return list(zip(freq,smooth)), max(freq), max(smooth)
        except Exception as e:
            Logger.error(f"FFT: {e}"); return None,0,0


if __name__ == "__main__":
    FFTApp().run()
