# buildozer.spec

[app]
# ── 앱 정보 ───────────────────────────────────────────────────────────────────
title               = FFTApp
package.name        = fftapp
package.domain      = org.test
version             = 0.1
orientation         = portrait
fullscreen          = 0

# ── 소스 포함 ────────────────────────────────────────────────────────────────
source.dir          = .
source.include_exts = py,kv,atlas,png,jpg,ttf,CSV

# ── 파이썬 요구사항 ──────────────────────────────────────────────────────────
# (크기 무시하고, CSV → FFT용으로만!)
requirements        = python3,kivy,numpy,plyer,android,requests

# ── Android 권한 및 설정 ─────────────────────────────────────────────────────
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET
android.api         = 33
android.minapi      = 21
android.ndk         = 23b
android.ndk_api     = 21
android.arch        = armeabi-v7a,arm64-v8a
# Gradle 사용 시 appcompat-v7 포함
android.gradle_dependencies = com.android.support:appcompat-v7:28.0.0

# ── 빌드 로깅 ────────────────────────────────────────────────────────────────
log_level           = 2
log_dir             = true

# ── python-for-android 커스텀 ─────────────────────────────────────────────────
p4a.branch          = master
p4a.bootstrap       = sdl2

[buildozer]
# 빌드 디렉터리
build_dir           = .buildozer
# 루트에서 실행 경고
warn_on_root        = 1
