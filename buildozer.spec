# buildozer.spec

[app]
# ── 앱 기본 정보 ───────────────────────────────────────────────────────────────
title               = FFTApp
package.name        = fftapp
package.domain      = org.test
version             = 0.1
orientation         = portrait
fullscreen          = 0

# ── 소스 포함 확장자 ───────────────────────────────────────────────────────────
source.dir          = .
source.include_exts = py,kv,atlas,png,jpg,ttf,CSV

# ── 파이썬 모듈 요구사항 ────────────────────────────────────────────────────────
# (코드에서 import 한 모든 모듈을 나열)
requirements        = python3,kivy,numpy,plyer,android

# ── Android 퍼미션 ─────────────────────────────────────────────────────────────
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET

# ── Android 플랫폼 & 툴체인 설정 ───────────────────────────────────────────────
android.api         = 33
android.minapi      = 21
android.arch        = armeabi-v7a,arm64-v8a
android.ndk         = 23b
android.ndk_api     = 21

# AppCompat 호환성 강화 (필요시)
android.gradle_dependencies = com.android.support:appcompat-v7:28.0.0

# 로그 설정
log_level           = 2
# 로그를 파일로 저장하려면 true
log_dir             = true
# logcat 필터 (필요시)
android.logcat_filters = *:S python:D

# p4a( python-for-android ) 설정
p4a.bootstrap       = sdl2
p4a.branch          = master


[app]
# … (기존 설정) …

# 반드시 추가: 설치·사용할 Build Tools 버전
android.build_tools_version = 33.0.2
# ── (선택) SDK/NDK 경로 고정 ─────────────────────────────────────────────────
# android.sdk_path   = /home/runner/android-sdk
# android.ndk_path   = /home/runner/android-sdk/ndk/23.1.7779620

[buildozer]
# 빌드 캐시 디렉터리
build_dir           = .buildozer
# APK 출력 디렉터리
bin_dir             = bin
# 루트 권한 경고
warn_on_root        = 1
