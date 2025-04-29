[app]
# 앱 기본 정보
title                = FFTApp
package.name         = fftapp
package.domain       = org.example
source.dir           = .
source.include_exts  = py,kv,csv
version              = 0.1
requirements         = python3,kivy,kivy_garden.graph,numpy,scipy
orientation          = portrait

# APK로 빌드 (AAB 사용 안 함)
android.use_aab      = False

[android]
# SDK/NDK/API 설정
android.api                 = 31
android.minapi              = 21
android.ndk_api             = 21
android.build_tools_version = 31.0.0

# 외부 SDK/NDK 절대 경로 (GitHub Actions 에서 동일하게 세팅)
android.sdk_path            = /home/runner/android-sdk
android.ndk_path            = /home/runner/android-sdk/ndk/25.2.9519653

# 라이선스 재확인 막기
android.accept_sdk_license  = True

# 권한 및 아키텍처
android.permissions         = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.archs               = arm64-v8a,armeabi-v7a

# Gradle bootstrap 사용
android.bootstrap           = gradle

# python-for-android는 pip 설치된 버전 사용
# [p4a] 섹션을 지정하지 않으면 buildozer가 pip 설치된 모듈을 그대로 사용합니다.
