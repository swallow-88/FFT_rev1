[app]
title             = FFTApp
package.name      = fftapp
package.domain    = org.example
source.dir        = .
source.include_exts = py,kv,csv
version           = 0.1
requirements      = python3,kivy,kivy_garden.graph,numpy,scipy
orientation       = portrait

android.use_aab = False

[android]
android.api                 = 31
android.minapi              = 21
android.ndk_api             = 21
android.build_tools_version = 31.0.0
# gradle bootstrap 권장
android.bootstrap           = gradle
android.permissions         = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.archs               = arm64-v8a,armeabi-v7a

android.ndk = 25.2.9519653

branch = v2023.9.16

# Buildozer/P4A 가 주입해 두길 바라는 절대경로
android.sdk_path = /home/runner/android-sdk
android.ndk_path = /home/runner/android-ndk-r25b      # ← 25b 를 받아 놓은 경우
android.accept_sdk_license = True                     # 라이선스 재확인 막기
# 이미 gradle bootstrap 을 쓰고 있으니 굳이 ant 1.9.4 가 필요 없다
p4a.bootstrap = gradle

[p4a]
# (str) python-for-android 포크 URL과 사용할 태그/브랜치
p4a.url    = https://github.com/kivy/python-for-android.git
p4a.branch = v2023.9.16
release = false

