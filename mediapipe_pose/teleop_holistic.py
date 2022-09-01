#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Taken and modified from
https://github.com/elggem/naoqi-pose-retargeting
(Apache 2.0 Licensed)
Part of this code is based on 
https://github.com/Kazuhito00/mediapipe-python-sample/blob/main/sample_pose.py
(Apache 2.0 Licensed)
"""

import copy
import argparse
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from time import time, sleep

from utils import CvFpsCalc
# from utils import KeypointsToAngles
from utils import SocketSend
from utils import HandsUtils
# from utils import SocketReceiveSignal
# from utils import calc_bounding_rect, draw_landmarks, plot_world_landmarks, draw_bounding_rect

from utils.face_geometry import(  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# uncomment next line to use all points for PnP algorithm
# points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];

frame_height, frame_width, channels = (720, 1280, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

dist_coeff = np.zeros((4, 1))

# keypointsToAngles = KeypointsToAngles()
hu = HandsUtils()

blue = (66, 135, 245)
dark_blue = (41, 82, 148)
orange = (224, 109, 63)
dark_orange = (158, 77, 44)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--fps", type=int, default=10)

    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')
    # parser.add_argument('--plot_angle_trace', action='store_true')
    
    parser.add_argument('--enable_teleop', action='store_true')

    args = parser.parse_args()

    return args


def socket_stream_landmarks(ss, landmarks, rHand_closed, lHand_closed, rHand_opened, lHand_opened, face_pose):
    p = []
    for landmark in landmarks.landmark:
        p.append([landmark.x, landmark.y, landmark.z])
    # p = np.array(p)

    pNeck =   (0.5 * (np.array(p[11]) + np.array(p[12]))).tolist()
    pMidHip = (0.5 * (np.array(p[23]) + np.array(p[24]))).tolist()
    
    wp_dict = {}

    wp_dict['0'] = p[0]    # Nose
    wp_dict['1'] = pNeck   # Neck
    
    wp_dict['2'] = p[11]   # LShoulder
    wp_dict['3'] = p[13]   # LElbow
    wp_dict['4'] = p[15]   # LWrist
    
    wp_dict['5'] = p[12]   # RShoulder
    wp_dict['6'] = p[14]   # RElbow
    wp_dict['7'] = p[16]   # RWrist
    
    wp_dict['8'] = pMidHip # MidHip
    
    wp_dict['9'] = rHand_closed
    wp_dict['10'] = lHand_closed
    wp_dict['11'] = rHand_opened
    wp_dict['12'] = lHand_opened
    wp_dict['13'] = face_pose[0]
    wp_dict['14'] = face_pose[1]
    
    ss.send(wp_dict)

def checkLim(val, limits):
    return val < limits[0] or val > limits[1]

# def do_teleop(landmarks):
#     p = []
#     for index, landmark in enumerate(landmarks.landmark):
#         p.append([landmark.x, landmark.y, landmark.z])
#     p = np.array(p)

#     limitsLShoulderPitch = [-2.0857, 2.0857]
#     limitsRShoulderPitch = [-2.0857, 2.0857]
#     limitsLShoulderRoll  = [ 0.0087, 1.5620]
#     limitsRShoulderRoll  = [-1.5620,-0.0087]
#     limitsLElbowYaw      = [-2.0857, 2.0857]
#     limitsRElbowYaw      = [-2.0857, 2.0857]
#     limitsLElbowRoll     = [-1.5620,-0.0087]
#     limitsRElbowRoll     = [ 0.0087, 1.5620]
#     limitsHipPitch       = [-1.0385, 1.0385]

#     pNeck =   (0.5 * (np.array(p[11]) + np.array(p[12]))).tolist()
#     pMidHip = (0.5 * (np.array(p[23]) + np.array(p[24]))).tolist()

#     LShoulderPitch, LShoulderRoll = keypointsToAngles.obtain_LShoulderPitchRoll_angles(pNeck, p[11], p[13], pMidHip)
#     RShoulderPitch, RShoulderRoll = keypointsToAngles.obtain_RShoulderPitchRoll_angles(pNeck, p[12], p[14], pMidHip)
    
#     LElbowYaw, LElbowRoll = keypointsToAngles.obtain_LElbowYawRoll_angle(pNeck, p[11], p[13], p[15])
#     RElbowYaw, RElbowRoll = keypointsToAngles.obtain_RElbowYawRoll_angle(pNeck, p[12], p[14], p[16])

#     HipPitch = keypointsToAngles.obtain_HipPitch_angles(pMidHip, pNeck) # This is switched, why?
#     # angles = [LShoulderPitch,LShoulderRoll, LElbowYaw, LElbowRoll, RShoulderPitch,RShoulderRoll, RElbowYaw, RElbowRoll, HipPitch]
    

#     # print(LShoulderPitch*180/np.pi, LShoulderRoll*180/np.pi)
#     # angle_trace.append(angles)

#     if (checkLim(LShoulderPitch, limitsLShoulderPitch) or 
#         checkLim(RShoulderPitch, limitsRShoulderPitch) or
#         checkLim(LShoulderRoll, limitsLShoulderRoll) or 
#         checkLim(RShoulderRoll, limitsRShoulderRoll) or
#         checkLim(LElbowYaw, limitsLElbowYaw) or 
#         checkLim(RElbowYaw, limitsRElbowYaw) or
#         checkLim(LElbowRoll, limitsLElbowRoll) or 
#         checkLim(RElbowRoll, limitsRElbowRoll)):
#         return False
#     else:
#         return True

def rotationMatrixToEulerAngles(R) :

    #assert(isRotationMatrix(R))
 
    #To prevent the Gimbal Lock it is possible to use
    #a threshold of 1e-6 for discrimination
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2]) 
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1]) 
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark
    # enable_teleop = args.enable_teleop
    enable_teleop = True

    video = args.video
    fps = args.fps

    if len(video) == 0:
        cap = cv.VideoCapture(cap_device)
    else:
        cap = cv.VideoCapture(video)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_holistic = mp.solutions.holistic
    pose = mp_holistic.Holistic(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if enable_teleop:
        # Initialize socket to send keypoints
        ss = SocketSend()
        # sr = SocketReceiveSignal()

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    if plot_world_landmark:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    # Init hands state
    rHand_closed = False
    lHand_closed = False
    rHand_opened = False
    lHand_opened = False
    rHand_angle = 0.0
    lHand_angle = 0.0
    euler_angles = [0.0, 0.0]
    
    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )
    
    try:
        while True:
            start = time()

            display_fps = cvFpsCalc.get()

            ret, image = cap.read()
            if not ret:
                break
            # image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            
            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    debug_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Draw hands landmarks
            if results.right_hand_landmarks:
                handedness = "Right"
                mp_drawing.draw_landmarks(debug_image,
                                            results.right_hand_landmarks,
                                            mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=dark_orange,
                                                                    thickness=2,
                                                                    circle_radius=4),
                                            mp_drawing.DrawingSpec(color=orange,
                                                                    thickness=2,
                                                                    circle_radius=2),
                                            )
                
                # Draw angles to image from joint list
                rHand_closed, rHand_opened, rHand_angle = hu.draw_finger_angles_3d(debug_image,
                                        results.right_hand_landmarks,
                                        hu.joint_list_low,
                                        handedness,
                                        cap_width,
                                        cap_height)
            
            if results.left_hand_landmarks:
                handedness = "Left"
                mp_drawing.draw_landmarks(debug_image,
                                            results.left_hand_landmarks,
                                            mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=dark_blue,
                                                                    thickness=2,
                                                                    circle_radius=4),
                                            mp_drawing.DrawingSpec(color=blue,
                                                                    thickness=2,
                                                                    circle_radius=2),)

                # Draw angles to image from joint list
                lHand_closed, lHand_opened, lHand_angle = hu.draw_finger_angles_3d(debug_image,
                                        results.left_hand_landmarks,
                                        hu.joint_list_low,
                                        handedness,
                                        cap_width,
                                        cap_height)
            
            if results.face_landmarks:
                face_landmarks = results.face_landmarks
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                landmarks = landmarks.T

                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )

                image_points = (
                    landmarks[0:2, points_idx].T
                    * np.array([frame_width, frame_height])[None, :]
                )
                model_points = metric_landmarks[0:3, points_idx].T

                # see here:
                # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
                pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
                mp_rotation_vector, _ = cv.Rodrigues(pose_transform_mat[:3, :3])
                mp_translation_vector = pose_transform_mat[:3, 3, None]

                euler_angles = rotationMatrixToEulerAngles(-pose_transform_mat[:3, :3])
                
                # print("Euler: ", euler_angles[:2])  
                
                # for face_landmarks in face_landmarks:
                # mp_drawing.draw_landmarks(
                #     image=debug_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=drawing_spec,
                #     connection_drawing_spec=drawing_spec,
                # )

                nose_tip = model_points[0]
                nose_tip_extended = 2.5 * model_points[0]
                (nose_pointer2D, jacobian) = cv.projectPoints(
                    np.array([nose_tip, nose_tip_extended]),
                    mp_rotation_vector,
                    mp_translation_vector,
                    camera_matrix,
                    dist_coeff,
                ) 
                
                nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
                debug_image = cv.line(
                    debug_image, nose_tip_2D, nose_tip_2D_extended, (0, 255, 0), 4
                )
            
            # if plot_world_landmark:
            #     if results.pose_world_landmarks is not None:
            #         plot_world_landmarks(plt, ax, results.pose_world_landmarks)

            if results.pose_world_landmarks is not None:
                cv.putText(debug_image, "TRACKING", (10, 90),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
                
                # if not do_teleop(results.pose_world_landmarks):
                #     # limit reached
                #     cv.putText(debug_image, "LIMIT", (10, 140),
                #         cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

                if enable_teleop:
                    socket_stream_landmarks(ss, results.pose_world_landmarks, rHand_closed, lHand_closed, rHand_opened, lHand_opened, euler_angles)

            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 145, 255), 2, cv.LINE_AA)

            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            
            image = cv.flip(debug_image, 1)
            cv.imshow('Pepper Teleop', debug_image)

            # if (len(video)>0):
            sleep(max(1./fps - (time() - start), 0))
            
    except KeyboardInterrupt:
        cap.release()
        cv.destroyAllWindows()
        
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()