# -*- encoding: UTF-8 -*-

import qi
# from naoqi import ALProxy
import argparse
import sys
import time
# import numpy as np
# import audioop
import threading
import math

import speech_recognition as sr
from threading import Thread

from head_motion import HeadMotionThread

import credentials

GOOGLE_CLOUD_SPEECH_CREDENTIALS = credentials.GOOGLE_CLOUD_SPEECH_CREDENTIALS
PREFERRED_PHRASES = ['stop talking', 'start talking', 'start moving', 'stop moving',
                     'move forward', 'go forward', 'move backwards', 'move back',
                     'go back', 'go backwards', 'move right', 'go right', 
                     'move left', 'go left', 'rotate left', 'turn left',
                     'rotate right', 'turn right', 'turn around', 'watch right',
                     'look right', 'watch left', 'look left', 'watch up', 
                     'look up', 'watch down', 'look down', 'watch ahead', 
                     'look ahead', 'track arm', 'follow arm', 'stop tracking',
                     'stop following', "track user", 'stop focus', 'call',
                     'skype call', 'skype', 'head control']

class SpeechThread(Thread):
    def __init__(self, session, q, q_rec, q_button):
        
        self.session = session
        self.text = None
        self.rec = False
        self.is_running = True
        self.q_text = q
        self.q_rec = q_rec
        
        self.q_button = q_button
        self.arm_tracking_event = threading.Event()
        self.stop_event = threading.Event()
        
        self.hmt = HeadMotionThread(self.session, self.arm_tracking_event, self.stop_event)
        self.hmt.start()
        
        # Speech recognizer  
        self.r = sr.Recognizer()
        self.r.energy_threshold = 1000
        self.r.dynamic_energy_threshold = True
        self.r.pause_threshold = 0.5
        
        # Call the Thread class's init function
        Thread.__init__(self)
        print("SpeechThread started")
        
        # Get the service ALTextToSpeech.
        self.tts = self.session.service("ALTextToSpeech")
        self.motion = self.session.service("ALMotion")
        self.life_service = self.session.service("ALAutonomousLife")
        self.blink_service = self.session.service("ALAutonomousBlinking")
    
    # Override the run() function of Thread class
    def run(self):
        # time to perform the movements
        t = 3
        # distance covered by every movement
        d = 0.5
        # angle of rotation 45 degrees
        angle = math.pi/4
            
        # main loop
        while self.is_running:
            
            # check for signals from the GUI
            if not self.q_rec.empty():
                command = self.q_rec.get(block=False, timeout= None)
                if command == "Rec":
                    self.rec = True
                elif command == "StopRec":
                    self.rec = False
                elif command == "StopRun":
                    self.rec = False
                    self.is_running = False
                    
            # Voice recognition
            if self.rec:
                self.text = self.recognize()
                if self.text is not None:

                    # text to lower case
                    txt = self.text.lower().strip()
                    print("Recognized phrase: "+ txt)
                    
                    try:      
                        # Voice commands to control Pepper position and the GUI
                        if 'move forward' in txt or 'go forward' in txt:
                            x = d
                            y = 0.0
                            theta = 0.0
                            self.motion.moveTo(x, y, theta, t)
                            
                        elif 'move backwards' in txt or 'go backwards' in txt or\
                             'move backward' in txt or 'go backward' in txt or\
                             'move back' in txt or 'go back' in txt:
                            x = -d
                            y = 0.0
                            theta = 0.0
                            self.motion.moveTo(x, y, theta, t)

                        elif 'move right' in txt or 'go right' in txt or\
                             'move to the right' in txt or 'go to the right' in txt:
                            x = 0.0
                            y = -d
                            theta = 0.0                         
                            self.motion.moveTo(x, y, theta, t)

                        elif 'move left' in txt or 'go left' in txt or\
                             'move to the left' in txt or 'go to the left' in txt:
                            x = 0.0
                            y = d
                            theta = 0.0
                            self.motion.moveTo(x, y, theta, t)
                            
                        elif 'rotate left' in txt or 'turn left' in txt:
                            x = 0.0
                            y = 0.0
                            
                            theta = angle    
                            self.motion.moveTo(x, y, theta, t)
                        
                        elif 'rotate right' in txt or 'turn right' in txt: 
                            x = 0.0
                            y = 0.0
                            theta = -angle
                            self.motion.moveTo(x, y, theta, t)
                        
                        elif 'turn around' in txt:
                            x = 0.0
                            y = 0.0
                            theta = math.pi
                            self.motion.moveTo(x, y, theta, t)
                            
                        elif 'stop talking' in txt:
                            self.q_button.put(txt)
                        
                        elif 'start pepper' in txt or 'start robot' in txt or 'start moving' in txt:
                            self.q_button.put('start pepper')
                        
                        elif 'stop pepper' in txt or 'stop robot' in txt or 'stop moving' in txt:
                            self.q_button.put('stop pepper') 
                        
                        elif 'watch right' in txt or 'look right' in txt or 'look to the right' in txt:
                            self.q_button.put('stop head')
                            self.life_service.stopAll()
                            # self.life_service.setState('disabled')
                            # stop arm tracking
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            
                            # stop user tracking
                            if self.life_service.getAutonomousAbilityEnabled("BasicAwareness"):
                                self.life_service.setAutonomousAbilityEnabled("BasicAwareness", False)
                            # move head right
                            self.motion.setStiffnesses("HeadYaw", 1)
                            # self.motion.setStiffnesses("HeadPitch", 1)
                            names = ['HeadYaw']
                            angles = [-angle]
                            self.motion.setAngles(names, angles, 0.15)   
                            
                              
                        elif 'watch left' in txt or 'look left' in txt or 'look to the left' in txt:
                            self.q_button.put('stop head')
                            self.life_service.stopAll()
                            # self.life_service.setState('disabled')
                            # stop arm tracking
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            # stop user tracking
                            if self.life_service.getAutonomousAbilityEnabled("BasicAwareness"):
                                self.life_service.setAutonomousAbilityEnabled("BasicAwareness", False)
                            # move head left
                            self.motion.setStiffnesses("HeadYaw", 1)
                            # self.motion.setStiffnesses("HeadPitch", 1)
                            names = ['HeadYaw']
                            angles = [angle]
                            self.motion.setAngles(names, angles, 0.15)   
                            
                            
                        elif 'watch up' in txt or 'look up' in txt:
                            self.q_button.put('stop head')
                            self.life_service.stopAll()
                            # self.life_service.setState('disabled')
                            # stop arm tracking
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            
                            # stop user tracking
                            if self.life_service.getAutonomousAbilityEnabled("BasicAwareness"):
                                self.life_service.setAutonomousAbilityEnabled("BasicAwareness", False)
                            # MOVE HEAD UP
                            self.motion.setStiffnesses("HeadPitch", 1)
                            # self.motion.setStiffnesses("HeadPitch", 1)
                            names = ['HeadPitch']
                            angles = [-angle/2]
                            self.motion.setAngles(names, angles, 0.15) 
                               
                            
                        elif 'watch down' in txt or 'look down' in txt:
                            self.q_button.put('stop head')
                            
                            self.life_service.stopAll()
                            # self.life_service.setState('disabled')
                            # stop arm tracking
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            # stop user tracking
                            if self.life_service.getAutonomousAbilityEnabled("BasicAwareness"):
                                self.life_service.setAutonomousAbilityEnabled("BasicAwareness", False)
                            # move head down
                            self.motion.setStiffnesses("HeadPitch", 1)
                            # self.motion.setStiffnesses("HeadPitch", 1)
                            names = ['HeadPitch']
                            angles = [angle/2]
                            self.motion.setAngles(names, angles, 0.15) 
                               
                        
                        elif 'watch ahead' in txt or 'look ahead' in txt or\
                             'look forward' in txt or 'watch forward' in txt:
                            self.q_button.put('stop head')
                            self.life_service.stopAll()
                            # self.life_service.setState('disabled')
                            # stop arm tracking
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            # stop user tracking
                            if self.life_service.getAutonomousAbilityEnabled("BasicAwareness"):
                                self.life_service.setAutonomousAbilityEnabled("BasicAwareness", False)
                                
                            # move head down
                            self.motion.setStiffnesses("HeadPitch", 1)
                            self.motion.setStiffnesses("HeadYaw", 1)
                            names = ['HeadYaw', 'HeadPitch']
                            angles = [0, 0]
                            self.motion.setAngles(names, angles, 0.15)  
                             
                            
                        elif 'track arm' in txt or 'follow arm' in txt or 'truck arm' in txt:
                            self.q_button.put('stop head')
                            self.text = "follow arm"
                            self.life_service.stopAll()
                            # self.life_service.setState('disabled')
                            if self.life_service.getAutonomousAbilityEnabled("BasicAwareness"):
                                self.life_service.setAutonomousAbilityEnabled("BasicAwareness", False)
                            if not self.arm_tracking_event.is_set():
                                self.arm_tracking_event.set()
                            
                            
                        elif 'stop tracking' in txt or 'stop following' in txt or\
                             'stop arm tracking' in txt or 'stop arm following' in txt:
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            
                        elif 'track user' in txt or 'truck user' in txt:
                            self.q_button.put('stop head')
                            if self.arm_tracking_event.is_set():
                                self.arm_tracking_event.clear()
                            self.life_service.setAutonomousAbilityEnabled("BasicAwareness", True)
                            
                        elif 'stop focus' in txt:
                            self.life_service.stopAll()
                            
                        elif 'call' in txt or 'skype call' in txt or 'skype' in txt:
                            self.q_button.put('call')
                        
                        elif 'head control' in txt:
                            self.q_button.put('head control')
                        else:
                            if self.session.isConnected():
                                # Repeat the recognized text
                                self.tts.say(self.text)
                            
                        # Put text in a queue for the GUI
                        self.q_text.put(self.text) 
                    except RuntimeError as e:
                        print(e)
                        print("Error:", self.text)
            else:
                if self.is_running: 
                    time.sleep(0.5)
                    
        # stop arm tracking thread
        self.arm_tracking_event.set()
        self.stop_event.set()     
        self.hmt.join()
          
        # self.q_text.put("Speech thread terminated correctly")            
        print("Speech thread terminated correctly")

        
    ## method recognize
    #
    #  Record voice from microphone and recognize it using Google Speech Recognition
    def recognize(self):
        # print("Energythresh speechThread:",self.r.energy_threshold)
        with sr.Microphone(device_index=1) as source:  
            recognized_text = None
            try:
                # Receive audio from microphone
                self.audio = self.r.listen(source, timeout=None, phrase_time_limit=3)

                # received audio data, recognize it using Google Speech Recognition
                # recognized_text = self.r.recognize_google(self.audio, language="en-US")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="130.251.13.113",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    

    # Parse arguments
    args = parser.parse_args()
    ip_addr = args.ip 
    port = args.port
    session = qi.Session()
    
    # Try to connect to the robot
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
            "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
      
    st = SpeechThread(session)
    st.start()
    
    st.join()
    