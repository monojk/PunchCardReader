# PunchCardReader
Android app to decode punch cards (IBM type, 12 rows, 80 rows) by taking a photo of the punch card. Best results if the back of the card on a black surface is photographed.

To build:
with: https://github.com/kivy/python-for-android/blob/develop/ci/makefiles/android.mk
export LEGACY_NDK=~/.android/android-ndk-legacy
otherwise the following error occurs:
Build failed: Please set the environment variable 'LEGACY_NDK' to point to a NDK location with gcc/gfortran support (supported NDK version: 'r21e')

export LEGACY_NDK=/home/joachim/.android/android-ndk-legacy
with: https://github.com/kivy/python-for-android/blob/develop/ci/makefiles/android.mk

Linux:
   time buildozer android debug
   buildozer -v android deploy run

Under Windows, use WSL  (WSL2). But must install also the windows version of adb
https://buildozer.readthedocs.io/en/latest/quickstart.html#run-my-application-from-windows-10

    buildozer android debug 2>&1 | tee build.log && buildozer -v android deploy run
    
Here is a screenshot of the app. Snapshot of the back of a punch card to get better results without disturbing text.      
![Picture of the app](PunchCardReader.jpg)
