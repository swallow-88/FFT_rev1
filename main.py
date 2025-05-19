# … 생략 …

class _P:          # 빈 Permission 더미
    READ_EXTERNAL_STORAGE = WRITE_EXTERNAL_STORAGE = ""
    READ_MEDIA_IMAGES = READ_MEDIA_AUDIO = READ_MEDIA_VIDEO = ""
    MANAGE_EXTERNAL_STORAGE = ""        # ← 추가

Permission = _P
# ------------------------------------------------------------------------

# ────────── UI ──────────
def build(self):
    # … 동일 …
    Clock.schedule_once(self._ask_perm, 0)
    return root

# ────────── 권한 확인 · 요청 (한 번만) ──────────
def _ask_perm(self, *_):
    if not ANDROID:
        self.btn_sel.disabled = False
        return

    need = [Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.MANAGE_EXTERNAL_STORAGE]
    if ANDROID_API >= 33:
        need += [Permission.READ_MEDIA_IMAGES,
                 Permission.READ_MEDIA_AUDIO,
                 Permission.READ_MEDIA_VIDEO]

    def _cb(perms, grants):
        if any(grants):
            self.btn_sel.disabled = False
        else:
            self.log("저장소 권한 거부 – CSV 파일을 열 수 없습니다")

    if all(check_permission(p) for p in need):
        self.btn_sel.disabled = False
    else:
        request_permissions(need, _cb)

# ────────── 파일 선택 (단일 정의만 남김) ──────────
def open_chooser(self, *_):
    # Android SAF picker → 실패 시 경로 기반 chooser 1회
    # Android 11+ 는 ‘모든 파일’ 허용 여부 먼저 체크
    # … (지금 코드의 두 번째 버전 그대로) …
