from heapq import heappush
from turtle import width
import mediapipe as mp
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandsUtils(object):
    
    def __init__(self) -> None:
        self.joint_list_up = [[8,7,6], [12,11,10], [16,15,14], [20,19,18], [4,3,2]]
        # self.joint_list_low = [[7,6,5], [11,10,9], [15,14,13], [19,18,17], [3,2,1]]
        self.joint_list_low = [[7,6,5], [11,10,9], [15,14,13], [19,18,17]]

    def get_label(self, index, hand, multi_handedness, width, height):
        output = None, None, None
        # print(len(multi_handedness))
        if len(multi_handedness) == 1:
            label = multi_handedness[0].classification[0].label
            score = multi_handedness[0].classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [width,height]).astype(int))
            
            output = text, coords, label
        else:
            for classification in multi_handedness:
                # print(multi_handedness)
                if classification.classification[0].index == index:
                    # print(multi_handedness)
                    # print(index, classification.classification[0].label, classification.classification[0].index)
                    # Process results
                    label = classification.classification[0].label
                    score = classification.classification[0].score
                    text = '{} {}'.format(label, round(score, 2))
                    
                    # Extract Coordinates
                    coords = tuple(np.multiply(
                        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                    [width,height]).astype(int))
                    
                    output = text, coords, label
                
        return output

    def calculate_angle_3d(self, a, b, c):  
        # Calculates angle between three points in 3d space     
        v1 = np.array([ a[0] - b[0], a[1] - b[1], a[2] - b[2] ])
        v2 = np.array([ c[0] - b[0], c[1] - b[1], c[2] - b[2] ])

        v1mag = np.sqrt([ v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2] ])
        v1norm = np.array([ v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag ])

        v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
        v2norm = np.array([ v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag ])
        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
        angle_rad = np.arccos(res)

        return angle_rad

    def draw_finger_angles_3d(self, image, hand, joint_list, rl, width, height):
        # Loop through hands
        # fingers_closed = []
        # fingers_opened = []
        finger_angles = []
        
        hand_closed = False
        hand_opened = False
        
        hand_angle = 0.0
        # print(joint_list)
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord
            
            angle = self.calculate_angle_3d(a, b, c)
            # print(f'{joint}: {math.degrees(angle)}')
            
            if angle > math.pi:
                angle = (2*math.pi) - angle
            
            # if angle < math.radians(120.0):
            #     fingers_closed.append(1)
            # if angle > math.radians(172.0):
            #     fingers_opened.append(1)
            
            finger_angles.append(angle)
            
            # cv2.putText(image, str(round(math.degrees(angle), 2)), tuple(np.multiply(b[0:2:], [width, height]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        hand_angle = sum(finger_angles)/len(finger_angles)
        if rl == 'Right':
                cv2.putText(image, f'{rl}: {math.degrees(hand_angle)}', (width-100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f'{rl}: {math.degrees(hand_angle)}', (width-100, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            
        if hand_angle < math.radians(100.0):
            hand_closed = True
            if rl == 'Right':
                cv2.putText(image, rl + " closed", (width-100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, rl + " closed", (width-100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        if hand_angle > math.radians(169.0):
            hand_opened = True
            if rl == 'Right':
                cv2.putText(image, rl + " opened", (width-100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, rl + " opened", (width-100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
        return hand_closed, hand_opened, hand_angle


# if __name__ == "__main__":
    
#     cap = cv2.VideoCapture(0)
#     width= 640
#     height = 480
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     hu = HandsUtils()

#     with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # BGR 2 RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Flip on horizontal
#             image = cv2.flip(image, 1)
            
#             # Set flag
#             image.flags.writeable = False
            
#             # Detections
#             results = hands.process(image)
            
#             # Set flag to true
#             image.flags.writeable = True
            
#             # RGB 2 BGR
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # Detections
#             # print(results)
#             hand_rl = None
#             # Rendering results
#             if results.multi_hand_landmarks:
#                 for num, hand in enumerate(results.multi_hand_landmarks):
#                     mp_drawing.draw_landmarks(image,
#                                               hand,
#                                               mp_hands.HAND_CONNECTIONS, 
#                                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                               mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
#                                              )
                    
#                     # Render left or right detection
#                     text, coord, hand_rl = hu.get_label(num, hand, results.multi_handedness, width, height)
#                     if text is not None:
#                         cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
#                     # Draw angles to image from joint list
#                     if hand_rl is not None:
#                         hu.draw_finger_angles_3d(image, hand, hu.joint_list_low, hand_rl, width, height)
              
#             # Save our image    
#             #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
#             cv2.imshow('Hands Tracking', image)

#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# '''
# def draw_finger_angles(image, results, joint_list):
#     # Loop through hands
#     fingers_closed = []
#     hand_closed = False
#     for hand in results.multi_hand_landmarks:
#         #Loop through joint sets 
#         for joint in joint_list:
#             a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
#             b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
#             c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
#             radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#             angle = np.abs(radians*180.0/np.pi)
            
#             if angle > 180.0:
#                 angle = 360-angle
            
#             if angle < 60.0:
#                 fingers_closed.append(1)

#             cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
#     if len(fingers_closed) > 3:
#         hand_closed = True
#         cv2.putText(image, "Closed", (10, 10),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
#     return hand_closed
# '''