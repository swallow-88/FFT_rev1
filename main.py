"""
step3_debug_fixed.py  –  Android/데스크탑 겸용 · SAF 테스트용 최소 예제
"""

# ────────────────────────────────────────────────────────────────
# 0)  공통 모듈 import
# ────────────────────────────────────────────────────────────────
import os, sys, traceback, urllib.parse, uuid, csv, threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.utils        import platform
from kivy.clock        import Clock
from kivy.logger       import Logger
from plyer             import filechooser

# ────────────────────────────────────────────────────────────────
# 1)  dummy android  ―  recipe 가 없을 때도 죽지 않도록
# ────────────────────────────────────────────────────────────────
try:
    import android              # recipe 가 있으면 OK
except ImportError:             # 없으면 즉시 더미 삽입
    import types
    android = types.ModuleType("android")
    android.activity = None     # plyer.filechooser 가 참조하는 속성
    sys.modules["android"] = android

# 이제부터는 언제든  from android import activity  해도 크래시 X
if platform == "android":
    from android import activity

# pyjnius 존재 여부는 나중에 try/except 로 확인
# ────────────────────────────────────────────────────────────────
# 2)  전역 크래시 → /sdcard/step3_crash.log + logcat
# ────────────────────────────────────────────────────────────────
def _ex_hook(et, ev, tb):
    txt = "".join(traceback.format_exception(et, ev, tb))
    try:
        with open("/sdcard/step3_crash.log", "a") as fp:
            fp.write(txt + "\n" + "="*70 + "\n")
    except Exception:
        pass
    Logger.error(txt)

sys.excepthook = _ex_hook

# ────────────────────────────────────────────────────────────────
# 3)  content:// (SAF) → 캐시 복사 or 파일 경로 변환
# ────────────────────────────────────────────────────────────────
def uri_to_file(source: str) -> str | None:
    """
    •  file://...          →  실제 경로
    •  /storage/...        →  그대로 (존재 여부만 체크)
    •  content://...       →  앱 cache 로 복사  (pyjnius 필요)
    실패 시 None
    """
    if not source:
        return None

    # file:// → 실제 경로 (URL-decode 포함)
    if source.startswith("file://"):
        real = urllib.parse.unquote(source[7:])
        return real if os.path.exists(real) else None

    # 전통 경로는 그대로
    if not source.startswith("content://"):
        return source if os.path.exists(source) else None

    # ===== SAF  복사 =====
    try:
        from jnius import autoclass, jarray    # pyjnius 필요
        act  = autoclass("org.kivy.android.PythonActivity").mActivity
        Uri  = autoclass("android.net.Uri")
        Cols = autoclass("android.provider.OpenableColumns")

        cr   = act.getContentResolver()
        uri  = Uri.parse(source)

        # 파일 이름 구하기
        name = "tmp"
        c = cr.query(uri, [Cols.DISPLAY_NAME], None, None, None)
        if c and c.moveToFirst():
            name = c.getString(0)
        if c:
            c.close()

        dst_path = os.path.join(
            act.getCacheDir().getAbsolutePath(),
            f"{uuid.uuid4().hex}-{name}"
        )

        istream = cr.openInputStream(uri)
        buf     = jarray('b')(8192)
        with open(dst_path, "wb") as out:
            while True:
                n = istream.read(buf)
                if n == -1: break
                out.write(bytes(buf[:n]))
        istream.close()
        Logger.info(f"SAF copy → {dst_path}")
        return dst_path

    except Exception as e:
        Logger.error(f"SAF copy fail: {e}")
        return None

# ────────────────────────────────────────────────────────────────
# 4)  Kivy UI  (파일 선택 결과만 표시)
# ────────────────────────────────────────────────────────────────
class DemoApp(App):
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.lbl = Label(text="Pick CSV"); root.add_widget(self.lbl)
        root.add_widget(Button(text="Select",
                               on_press=self.open_picker))
        return root

    def open_picker(self, *_):
        # native=True  ➜ SAF 파일 피커,   False ➜ 전통 경로 피커
        filechooser.open_file(self.on_selected,
                              multiple=True,
                              filters=[("CSV","*.csv")],
                              native=True)

    def on_selected(self, selection):
        Logger.info(f"PICK ⇒ {selection}")
        if not selection:
            self.lbl.text = "취소됨"; return

        results = []
        for s in selection:
            real = uri_to_file(s)
            results.append(real)

        self.lbl.text = "\n".join(str(p) for p in results)

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DemoApp().run()
