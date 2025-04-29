[app]
title                   = FFTApp
package.name            = fftapp
package.domain          = org.example
source.dir              = .
source.include_exts     = py,kv,csv
version                 = 0.1
requirements            = python3,kivy,kivy_garden.graph,numpy,scipy
orientation             = portrait
android.use_aab         = False

[android]
# SDK/NDK/API 설정
android.api             = 31
android.minapi          = 21
android.ndk_api         = 21
android.build_tools_version = 31.0.0
android.ndk             = 25.2.9519653

# 권장: gradle bootstrap
android.bootstrap       = gradle
p4a.bootstrap           = gradle

android.permissions     = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.archs           = arm64-v8a,armeabi-v7a

# 외부에서 설치해 둔 경로
android.sdk_path        = /home/runner/android-sdk
android.ndk_path        = /home/runner/android-ndk-r25b
android.accept_sdk_license = True

[p4a]
# pip으로 설치한 버전을 그대로 쓸 거면 이 줄들 통째로 삭제
p4a.url                 = https://github.com/kivy/python-for-android.git
p4a.branch              = v2024.1.21
release                 = false
