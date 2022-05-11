from threading import Thread
import time

class RepeatHeadCommands(Thread):
    def __init__(self, session, q_head_pose, e_pause, e_stop):
        self.session = session
        self.q_head_pose = q_head_pose
        self.e_pause = e_pause
        self.e_stop = e_stop
        
        # Call the Thread class's init function
        Thread.__init__(self)
        print("RepeatHeadCommands thread started")

        
    def run(self):
        names = ["HeadYaw", "HeadPitch"]
        angles = [0.0, 0.0]
        motion_service  = self.session.service("ALMotion")
        # counter = 0
        while not self.e_stop.is_set(): 
            if self.e_pause.is_set():
                if not self.q_head_pose.empty():
                    # print(counter)
                    # counter = 0
                    angles = self.q_head_pose.get(block=False, timeout=None)
                    # print("repeat New Angles", angles, time.time())
                    # motion_service.setAngles(names, angles, 0.1)
                    
                # counter += 1
                motion_service.setAngles(names, angles, 0.2)
            else:
                time.sleep(0.1)
        
        print("RepeatHeadCommands thread terminated correctly")

