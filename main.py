"""
step3_debug.py  –  “Android 코드 한 줄씩 열어 보기” 전용 최소 예제
"""
import os, sys, traceback, urllib.parse
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button   import Button
from kivy.uix.label    import Label
from kivy.utils        import platform
from plyer             import filechooser
from kivy.logger       import Logger
from kivy.clock        import Clock

# ──────────────────────────────  전역 크래시 → logcat + /sdcard
def _ex(et, ev, tb):
    txt = "".join(traceback.format_exception(et, ev, tb))
    try: open("/sdcard/step3_crash.log", "a").write(txt)
    except: pass
    Logger.error(txt)
sys.excepthook = _ex

ANDROID = platform == "android"             # 데스크탑 테스트도 OK

# ──────────────────────────────  SAF / 전통 경로 처리
def uri_to_file(p: str) -> str | None:
    """
    p 가…
      • file://…  → 실경로 반환
      • /storage/emulated/…  → 그대로
      • content://… → (안드로이드 블록을 열면) cache 로 복사
    실패 시 None
    """
    if not p:
        return None

    # ─── file:// → 실제 경로 ───────────────────────────
    if p.startswith("file://"):
        real = urllib.parse.unquote(p[7:])
        return real if os.path.exists(real) else None

    # ─── 전통 경로 (/storage/…) ────────────────────────
    if not p.startswith("content://"):
        return p if os.path.exists(p) else None

    # ------------------------------------------------------------------
    # ▼▼▼ 아래 블록을 ‘한 줄씩’ 주석 해제해 가며 테스트하세요 ▼▼▼
    # ------------------------------------------------------------------
    
    try:
        Logger.info("PYDBG 0  – jnius import 시도")
        from jnius import autoclass, jarray            # ← ①
        Logger.info("PYDBG 1  – jnius import 성공")

        act  = autoclass("org.kivy.android.PythonActivity").mActivity   # ← ②
        Uri  = autoclass("android.net.Uri")                             # ← ③
        #Cols = autoclass("android.provider.OpenableColumns")            # ← ④
        cr   = act.getContentResolver()

        uri  = Uri.parse(p)
        name = "tmp"
        c = cr.query(uri, [Cols.DISPLAY_NAME], None, None, None)
        if c and c.moveToFirst():
            name = c.getString(0)
        if c: c.close()

        Logger.info(f"PYDBG 2  – SAF filename = {name}")

        ist = cr.openInputStream(uri)
        dst = os.path.join(
            act.getCacheDir().getAbsolutePath(),
            f"{uuid.uuid4().hex}-{name}"
        )

        #buf = jarray('b')(8192)                       # ← ⑤
        with open(dst, "wb") as out:
            while True:
                n = ist.read(buf)
                if n == -1: break
                out.write(bytes(buf[:n]))
        ist.close()
        Logger.info(f"PYDBG 3  – 복사 완료 {dst}")
        return dst
    except Exception as e:
        Logger.error(f"PYDBG ERR – SAF copy fail {e}")
        return None
    
    # ------------------------------------------------------------------
    Logger.info("PYDBG –  content:// but Android block 아직 OFF")
    return None
# ──────────────────────────────  Kivy UI
class Demo(App):
    def build(self):
        root = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.lbl = Label(text="pick csv"); root.add_widget(self.lbl)
        root.add_widget(Button(text="Select", on_press=self.pick))
        return root

    def pick(self,*_):
        filechooser.open_file(self.on_pick, multiple=True,
                              filters=[("CSV","*.csv")],
                              native=True)          # SAF 사용

    def on_pick(self, sel):
        Logger.info(f"PYDBG pick → {sel}")
        if not sel:
            self.lbl.text = "취소됨"; return
        paths=[]
        for raw in sel:
            real = uri_to_file(raw)
            paths.append(real)
        self.lbl.text = "\n".join(str(p) for p in paths)

if __name__=="__main__":
    Demo().run()
