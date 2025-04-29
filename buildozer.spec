[app]
# 기본 앱 정보
title                = FFTApp
package.name         = fftapp
package.domain       = org.example
source.dir           = .
source.include_exts  = py,kv,csv
version              = 0.1
requirements         = python3,kivy,kivy_garden.graph,numpy,scipy
orientation          = portrait

# 로그 레벨 및 경고 설정
log_level            = 2
warn_on_root         = 1

[buildozer]
# Android 빌드 타겟
target                = android

[android]
# Android API / NDK 설정
android.api           = 31
android.minapi        = 21
android.ndk_api        = 21

# Gradle 부트스트랩 사용 (ANT 제거)
android.bootstrap     = gradle

# 지원 아키텍처
android.archs         = arm64-v8a,armeabi-v7a

# 외부에 설치된 SDK / NDK 경로
android.sdk_path      = /home/runner/android-sdk
android.ndk_path      = /home/runner/android-sdk/ndk/25.2.9519653

# 빌드 도구 버전 고정
android.build_tools_version = 31.0.0

# 권한
android.permissions   = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# python-for-android는 pip으로 설치된 버전 사용
# -> p4a.url / p4a.branch 설정하지 않음
