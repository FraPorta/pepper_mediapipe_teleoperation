# -*- encoding: UTF-8 -*-

import qi
from naoqi import ALProxy
import argparse
import sys
import time
import numpy as np
import audioop

import speech_recognition as sr
from threading import Thread

GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""{
  "type": "service_account",
  "project_id": "ordinal-algebra-316914",
  "private_key_id": "23a8d985f6fa1a4e2b93c5396c201a99de5eb138",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDjXLx6u67BjZoM\nfoMbr4FgGQIs7rgmx8qFkL1kw9UWllJI+oUv1hmZBEZc/c9/7Q1Q1WNrgke8jItc\neZKsgxCz3sfvYbYK2XtbApWjRndTrdHI6eC08mU/Z1WplS+i2i1EAgCs1Spy+lxs\nZDBroU966jrjCdPm0Orps9jvwdaHWtUxBTLnphx0nKb8XOAf2kPCAhyRPl0rYTwJ\nTIVkb/HFx5xQG3bDs6AK/tuJmF5/cYQv+cYov4Y/mQz4dzF7P1Kq9LxcKBQ+D2e+\n31ErK7tyJZQuTkQ+XkCVl+UaVIhx9kETrwJ7uJOZB042fouX0s7ynbRhSwenIp9N\nOl6kuoQJAgMBAAECggEAFDtLk1FRpOwMjfmRAZztsocvLB/m9552LUn6rHWHp+2i\n71cJlH7lArqAZ2R3ey71LSD76pRqfUL1YLp5vIuiKBtWJ71GZXNCWJAjkCakVigk\nv4/ePMPIEisIEqHRnhhziDZ0aGzEjPwtzsBgladCSxH+QPc+Ka8kBD1Ke9VAGYWm\n886x1bFuGMBCJAxH9oUFGiwXQWaZeBGrf/gI5UCS3L5Y9eYcAsLBPIFaeOxfcjgo\np2evrUsQYDv8aud1umFkdGwexcACZxtZHlbZRMizqJcZa/TuVdChLFnLSkNvDlxJ\nqCPpXwBEXQL6i2KN12K2+yDw1A6X6i/59jqjrjORIQKBgQD7HdBzz43VyiQ/M1/v\nI69G6DA+a377C3D4EMvBFGM7TJIFPH5KBvSXIwmXKcDI48RzIGqy6dUuyGD5Lspv\nOx32aoxFCSCypqxPOxVcB7vN7KH82LVIWHXdvdRTVdwKpnOLRB2fMfl/3bhD+Mcn\n7deKqBiYxXORa0WNC9XWGk/ekQKBgQDnyKlSRA7CbWSlLzfUXI6Qak+0GAkz7+Lw\nhrU1YWEtgooSubVjZUHwCaJ0h2aTYFJkA3YMyGGX1uZmKqDNRpxkBMUU4O0wh9jo\nOBxe4coj1QBWMuRat5YZu3DAFoFpGGSmiaABWGbyk6So7Q+lg1lgzznWFgMeiTdJ\nDXflbJD5+QKBgANGrUyFfa03WOe2LN860POU3eRClMKDsLKbzXy0XmMMqa8FzgP7\nOT+rhlmBgvFb/1kdKCczY8Obe6BkmuF+nPJ1C2VvLA0InaDe/XVY6HtURfy9ewf5\nl3gQHPAFY7yD1WgQuG42QbIIW3oPidGcib5WWRPNJ6fTBXJEoEKNz1LxAoGABiSq\nYDTPk1Wk6j/jIezGFovKywIsFEVSZX+cg+qt0a/5CuADL7w6UCthM+d2z4cpB9+T\nnGkfNRAMET8l/erWMEx0EMaZYsTm+diq39TqL6LwnFhC9yiQgOQX6+9sxKVR3Zwe\nMoruR5WkMpn95SwjDU7QwJzavk9yuKvzto/3E7kCgYBN2YeEIwKkratuEtMxoTau\n4ApZBneQdk25wwWgRE/4HvYL0WfmfMjmOR2ybPb7UaGPT8ecEFreu2NM8cpv+fvS\n0/luhdtKwshjstbQ8vU0zvKs48RDl4IK8ySefksXFAQuVA82KzQ6SD8Ns0AjCrVA\nFa4sODGKcQd/ZwR/kqxqCQ==\n-----END PRIVATE KEY-----\n",
  "client_email": "fra-365@ordinal-algebra-316914.iam.gserviceaccount.com",
  "client_id": "112813127655483607587",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fra-365%40ordinal-algebra-316914.iam.gserviceaccount.com"
}"""

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
        
        # Call the Thread class's init function
        Thread.__init__(self)
        
        print("OkPepperThread started")
    
    # Override the run() function of Thread class
    def run(self):
        # self.r.adjust_for_ambient_noise(source, duration=0.5)  # listen for 0.5 second to calibrate the energy threshold for ambient noise levels
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
                    txt = self.text.lower()
                    print(txt)
                    
                    # Voice commands for the buttons
                    if txt == 'connect':
                        self.q_button.put(txt)
                    elif txt == 'start talking':
                        self.q_button.put(txt)
                    elif txt == 'start pepper' or txt == 'start robot' or txt == 'start moving':
                        self.q_button.put('start pepper')
                    elif txt == 'stop robot' or txt == 'stop pepper' or txt == 'stop moving':
                        self.q_button.put('stop pepper')

            else:
                if self.is_running: 
                    time.sleep(0.5)   
                 
        print("OkPepper thread terminated correctly")
        
    ## method recognize
    #
    #  Record voice from microphone and recognize it using Google Speech Recognition
    def recognize(self):
        # print(self.list_working_microphones())
        with sr.Microphone(device_index=1) as source:  
            
            recognized_text = None
            try:
                # Receive audio from microphone
                self.audio = self.r.listen(source, timeout=None, phrase_time_limit=None)
                
                # received audio data, recognize it using Google Speech Recognition
                # recognized_text = self.r.recognize_google(self.audio, language="en-EN")
                recognized_text = self.r.recognize_google_cloud(self.audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

        return recognized_text
        
        
# Main initialization
if __name__ == "__main__":
    
    opt = OkPepperThread()
    opt.start()
    
    opt.join()
    print('Thread finished cleanly')