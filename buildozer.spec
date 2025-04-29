# buildozer.spec

[app]
title               = FFTApp
package.name        = fftapp
package.domain      = org.test
version             = 0.1
orientation         = portrait
fullscreen          = 0

source.dir          = .
source.include_exts = py,kv,atlas,png,jpg,ttf,CSV

requirements        = python3,kivy,numpy,plyer,android,requests

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET
android.api         = 33
android.minapi      = 21
android.ndk         = 23b
android.ndk_api     = 21
android.arch        = armeabi-v7a,arm64-v8a
android.gradle_dependencies = com.android.support:appcompat-v7:28.0.0

log_level           = 2
log_dir             = true

p4a.branch          = master
p4a.bootstrap       = sdl2

[buildozer]
build_dir           = .buildozer
warn_on_root        = 1
