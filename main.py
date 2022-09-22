"""
Camera Example
==============
https://kivy.org/doc/stable/examples/gen__camera__main__py.html
This example demonstrates a simple use of the camera. It shows a window with
a buttoned labelled 'play' to turn the camera on and off. Note that
not finding a camera, perhaps because gstreamer is not installed, will
throw an exception during the kv language processing.

"""

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
import time
from android.permissions import request_permissions, Permission
import readPunchCard

# Uncomment these lines to see all the messages
# from kivy.logger import Logger
# import logging
# Logger.setLevel(logging.__ALL__)

Window.maximize()

Builder.load_string('''
#:import Clipboard kivy.core.clipboard.Clipboard
<CameraClick>:
    result: result
    start: start
    copy: copy
    quit: quit 
    orientation: 'horizontal'
    
    BoxLayout:
        orientation: 'vertical'
        size_hint: 0.9, 1

        Camera:
            id: camera
            size_hint: 1, 0.9
            #resolution: (900, 500)      # distorted
            #resolution: (800, 400)     # ok
            # 4032, 2268 = Galaxy S7, S8 = 9,1MP = 16:9
            #resolution: (4032, 2268)    # S7 ok, S8 crash    
            resolution: (1920, 1080)    # S8 ok
            allow_stretch: True
            keep_ratio: True
            play: True
            
        #TextInput:
        Label:
            id: result
            #multiline: False 
            size_hint: 1, 0.1
            height: '40dp'
            text: ''
            markup: True
            background_color: 0.91, 0.83, 0.78, 1
            # Default the background color for this label
            # to r 0, g 0, b 0, a 0   
                     
    BoxLayout:
        orientation: 'vertical'
        size_hint: 0.1, 1
        height: '40dp'
        
        Button: 
            id: quit      
            text: 'Quit'
            background_color: .8, .0, .0, 1
            on_press: app.stop()  

        # https://stackoverflow.com/questions/63790475/how-to-program-copy-to-paste-from-clipboard-buttons-on-app-developed-by-kivy            
        Button: 
            id: copy       
            text: 'Copy'
            background_color: 0, .0, .8, 1
            on_press: Clipboard.copy(root.plainText)       
        
        Button: 
            id: start       
            text: 'Scan'
            background_color: 0, .8, .0, 1
            on_press: root.capture() 
''')


class CameraClick(BoxLayout):

    resultText = ''
    plainText = ''

    def capture(self):
        """
        Function to capture the images and give them the names
        according to their captured time and date.
        """
        self.result.text = "Running"
        camera = self.ids['camera']
        fn = "/sdcard/CARD_{}.png".format(time.strftime("%Y%m%d_%H%M%S"))
        # camera.export_to_png(fn)
        camera.texture.save(fn)         # https://github.com/kivy/kivy/issues/7872
        print(f"Captured {fn}")
        self.result.text = "Running"
        self.resultText, self.plainText = readPunchCard.run(file=fn, contrast=1.3)
        self.result.text = self.resultText

    # https://gist.github.com/kived/0f450e738bf79c003253
    # https://stackoverflow.com/questions/38983649/kivy-android-share-image
    # https://stackoverflow.com/questions/63322944/how-to-use-share-button-to-share-content-of-my-app
    def shareText(self):
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Intent = autoclass('android.content.Intent')
        String = autoclass('java.lang.String')
        shareIntent = Intent(Intent.ACTION_SEND)
        shareIntent.setType('text/plain')
        shareIntent.putExtra(Intent.EXTRA_TEXT, self.plainText)

        chooser = Intent.createChooser(shareIntent, String('Share Punch Card Text'))
        PythonActivity.mActivity.startActivity(chooser)

        # from jnius import cast
        # currentActivity = cast('android.app.Activity', PythonActivity.mActivity)
        # currentActivity.startActivity(shareIntent)


class CardCamera(App):

    def build(self):
        request_permissions([
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE
        ])
        return CameraClick()


CardCamera().run()
