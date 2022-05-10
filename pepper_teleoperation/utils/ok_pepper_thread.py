# -*- encoding: UTF-8 -*-

# import qi
# from naoqi import ALProxy
# import argparse
# import sys
import time
# import numpy as np
# import audioop

import speech_recognition as sr
from threading import Thread
import credentials

GOOGLE_CLOUD_SPEECH_CREDENTIALS = credentials.GOOGLE_CLOUD_SPEECH_CREDENTIALS
PREFERRED_PHRASES = ['connect', 'start talking', 'start moving', 'stop moving', 'call', 'skype call', 'skype']

class OkPepperThread(Thread):
    def __init__(self, q_button, q_stop):
        # self.session = session
        self.text = None
        self.rec = False
        self.is_running = True
        
        self.q_button = q_button
        self.q_stop = q_stop
        
        # Speech recognizer  
        self.r = sr.Recognizer()
        self.r.energy_threshold = 1000
        self.r.dynamic_energy_threshold = True
        self.r.pause_threshold = 0.5
        
        # Call the Thread class's init function
        Thread.__init__(self)
        
        print("OkPepperThread started")
    
    # Override the run() function of Thread class
    def run(self):
        while self.is_running:
            if not self.q_stop.empty():
                command = self.q_stop.get(block=False, timeout= None)
                if command == "Rec":
                    self.rec = True
                elif command == "StopRec":
                    print('Stopped recording OkPepper')
                    self.rec = False
                elif command == "StopRun":
                    self.rec = False
                    self.is_running = False
                    
            if self.rec:
                self.text = self.recognize()
                if self.text is not None:
                    # text to lower case
                    txt = self.text.lower().strip()
                    print("Recognized phrase: "+ txt)
                    
                    # Voice commands for the buttons
                    if 'connect' in txt:
                        self.q_button.put(txt)
                    elif 'start talking' in txt:
                        self.q_button.put(txt)
                    elif 'start pepper' in txt or 'start robot' in txt or 'start moving' in txt:
                        self.q_button.put('start pepper')
                    elif 'stop robot' in txt or 'stop pepper' in txt or 'stop moving' in txt:
                        self.q_button.put('stop pepper')
                    elif 'call' in txt or 'skype call' in txt or 'skype' in txt:
                        self.q_button.put('call')

            else:
                if self.is_running: 
                    time.sleep(0.5)   
                 
        print("OkPepper thread terminated correctly")
        
    ## method recognize
    #
    #  Record voice from microphone and recognize it using Google Speech Recognition
    def recognize(self):
        # print("Energythresh OKPEpper:",self.r.energy_threshold)
        with sr.Microphone(device_index=1) as source:  
            
            recognized_text = None
            try:
                # Receive audio from microphone
                self.audio = self.r.listen(source, timeout=None, phrase_time_limit=2)
                
                # received audio data, recognize it using Google Speech Recognition
                recognized_text = self.r.recognize_google_cloud(self.audio,
                                                                preferred_phrases=PREFERRED_PHRASES,
                                                                credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
            # except Exception as e:
            #     print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return recognized_text
        
        
# Main initialization
if __name__ == "__main__":
    
    opt = OkPepperThread()
    opt.start()
    
    opt.join()
    print('Thread finished cleanly')