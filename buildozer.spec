# buildozer.spec

[app]

buildozer android clean   # 캐시-dist 초기화 (권장)
buildozer -v android debug

# ── 앱 기본 정보 ─────────────────────────────────────────────────────────────
title               = FFTApp
package.name        = fftapp
package.domain      = org.test
version             = 0.1
orientation         = portrait
fullscreen          = 0

# ── 소스 포함 확장자 ─────────────────────────────────────────────────────────
source.dir          = .
source.include_exts = py,kv,atlas,png,jpg,ttf,CSV



requirements = python3,kivy,plyer,pyjnius,numpy,android

#androidstorage4kivy@https://github.com/kivy-garden/androidstorage4kivy/archive/master.zip

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,READ_MEDIA_AUDIO,READ_MEDIA_VIDEO




# ── Android 플랫폼 & 툴체인 설정 ───────────────────────────────────────────────
android.api                 = 33
android.minapi              = 24
android.archs                = armeabi-v7a,arm64-v8a
android.ndk = 25.2.9519653
android.ndk_api             = 24

# ── Build Tools 버전 고정 ──────────────────────────────────────────────────────
android.build_tools_version = 33.0.2

# ── 시스템 SDK/NDK 경로 지정 ─────────────────────────────────────────────────
android.sdk_path            = /usr/local/lib/android/sdk
# r25b를 설치했다면 경로 예시:
android.ndk_path            = /usr/local/lib/android/sdk/ndk/25.2.9519653

# ── AppCompat 호환성 강화 (필요 시) ────────────────────────────────────────────
android.gradle_dependencies = com.android.support:appcompat-v7:28.0.0

# ── 로그 설정 ────────────────────────────────────────────────────────────────
log_level                   = 2
log_dir                     = true
android.logcat_filters      = *:S python:D

# ── python-for-android 설정 ──────────────────────────────────────────────────
p4a.bootstrap               = sdl2
#p4a.extra_args = --service androidstorage


#p4a.branch = stable

# python-for-android develop 브랜치를 clone 하지 않도록 주석 처리!
#p4a.branch                  = develop

    # … 나머지 설정 …



# ── (선택) SDK/NDK 경로 고정 ───────────────────────────────────────────────────
# android.sdk_path          = /home/runner/android-sdk
# android.ndk_path          = /home/runner/android-sdk/ndk/23.1.7779620

# Release 아티팩트 타입을 apk로 강제
android.release_artifact_types = apk


# lib2to3, test 모듈을 컴파일 대상에서 제외
#p4a.extra_args = --blacklist-requirements=test,lib2to3


# … 기존 설정 …
# p4a에 setup.py 실행을 요청합니다.

[buildozer]
# ── 빌드 캐시 및 출력 디렉터리 ───────────────────────────────────────────────
build_dir           = .buildozer
bin_dir             = bin

# ── 루트에서 실행 경고 ───────────────────────────────────────────────────────
warn_on_root        = 1
