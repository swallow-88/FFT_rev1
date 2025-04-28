import os
import numpy as np
from scipy.fft import rfft, rfftfreq
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup

KV = '''
<Root>:
    orientation: "vertical"
    padding: dp(10)
    spacing: dp(10)

    BoxLayout:
        size_hint_y: None
        height: "40dp"
        spacing: dp(10)

        Button:
            text: "CSV 1 선택"
            on_release: root.pick_file(1)
        Label:
            text: root.file1_name or "없음"

    BoxLayout:
        size_hint_y: None
        height: "40dp"
        spacing: dp(10)

        Button:
            text: "CSV 2 선택"
            on_release: root.pick_file(2)
        Label:
            text: root.file2_name or "없음"

    Button:
        size_hint_y: None
        height: "50dp"
        text: "분석 시작"
        on_release: root.run_fft()

    Graph:
        id: graph
        xlabel: "Frequency (Hz)"
        ylabel: "Amplitude"
        x_grid: True
        y_grid: True
        xmin: 0
        xmax: 100
        ymin: 0
        ymax: 1
'''

class Root(BoxLayout):
    file1_name = StringProperty("")
    file2_name = StringProperty("")
    file1_path = ""
    file2_path = ""

    def pick_file(self, idx):
        chooser = FileChooserIconView(filters=['*.csv'])
        popup = Popup(title="CSV 파일 선택", content=chooser, size_hint=(0.9, 0.9))
        chooser.bind(on_submit=lambda c, sel, *_: self._file_chosen(idx, sel, popup))
        popup.open()

    def _file_chosen(self, idx, selection, popup):
        if selection:
            path = selection[0]
            if idx == 1:
                self.file1_path = path
                self.file1_name = os.path.basename(path)
            else:
                self.file2_path = path
                self.file2_name = os.path.basename(path)
        popup.dismiss()

    def run_fft(self):
        if not (self.file1_path and self.file2_path):
            print("CSV 파일 두 개를 모두 선택하세요.")
            return

        # CSV → 1열만 로드
        data1 = np.loadtxt(self.file1_path, delimiter=',')[:, 0]
        data2 = np.loadtxt(self.file2_path, delimiter=',')[:, 0]

        n = min(len(data1), len(data2))
        data1, data2 = data1[:n], data2[:n]

        fs = 200  # 샘플링 주파수 예시(Hz) – 필요하면 CSV에 맞게 조정
        freq = rfftfreq(n, d=1 / fs)
        amp1 = np.abs(rfft(data1)) / n
        amp2 = np.abs(rfft(data2)) / n
        diff = np.abs(amp1 - amp2)

        g: Graph = self.ids.graph
        g.clear_plots()

        plot1 = MeshLinePlot(color=[0, 1, 0, 1])  # green
        plot2 = MeshLinePlot(color=[1, 0, 0, 1])  # red
        plot3 = MeshLinePlot(color=[1, 1, 1, 1])  # white

        # 100 Hz까지만
        mask = freq <= 100
        xs = freq[mask]
        plot1.points = list(zip(xs, amp1[mask]))
        plot2.points = list(zip(xs, amp2[mask]))
        plot3.points = list(zip(xs, diff[mask]))

        g.xmax = 100
        g.ymax = max(amp1.max(), amp2.max()) * 1.1
        g.add_plot(plot1)
        g.add_plot(plot2)
        g.add_plot(plot3)
        print("FFT 분석 완료.")

class FFTApp(App):
    def build(self):
        return Builder.load_string(KV)

if __name__ == "__main__":
    FFTApp().run()
