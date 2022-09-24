# PunchCardReader
Android kivy app to decode punch cards (IBM type, 12 rows, 80 rows) by taking a photo of the punch card. Best results if the back of the card on a black surface is photographed.

For Kivy see: https://kivy.org/

To build use buildozer: https://buildozer.readthedocs.io/en/latest/index.html

with: https://github.com/kivy/python-for-android/blob/develop/ci/makefiles/android.mk
    
    export LEGACY_NDK=~/.android/android-ndk-legacy

otherwise the following error occurs:

Build failed: Please set the environment variable 'LEGACY_NDK' to point to a NDK location with gcc/gfortran support (supported NDK version: 'r21e')

Note: this app uses the 8.4. version of Pillow (PIL) due to build recipes for kivy/python-for-android. Could not use PIL version 9.2.0

Linux:

    time buildozer android debug
    buildozer -v android deploy run

Under Windows, use WSL  (WSL2). But must install also the windows version of adb
https://buildozer.readthedocs.io/en/latest/quickstart.html#run-my-application-from-windows-10

    buildozer android debug 2>&1 | tee build.log && buildozer -v android deploy run
    
Here is a screenshot of the app. Snapshot of the back of a punch card to get better results without disturbing text.      
![Picture of the app](PunchCardReader.jpg)
