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
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from time import time, sleep

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from utils import CvFpsCalc
from utils import KeypointsToAngles
from utils import SocketSend
from utils import HandsUtils
from utils import SocketReceiveSignal
from utils import calc_bounding_rect, draw_landmarks, plot_world_landmarks, draw_bounding_rect

keypointsToAngles = KeypointsToAngles()
hu = HandsUtils()
angle_trace = []

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--fps", type=int, default=10)

    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.8)
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


def socket_stream_landmarks(ss, landmarks):
    p = []
    for index, landmark in enumerate(landmarks.landmark):
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

    # print(wp_dict)
    ss.send(wp_dict)

def checkLim(val, limits):
    return val < limits[0] or val > limits[1]

def do_teleop(landmarks):
    global angle_trace
    p = []
    for index, landmark in enumerate(landmarks.landmark):
        p.append([landmark.x, landmark.y, landmark.z])
    p = np.array(p)

    limitsLShoulderPitch = [-2.0857, 2.0857]
    limitsRShoulderPitch = [-2.0857, 2.0857]
    limitsLShoulderRoll  = [ 0.0087, 1.5620]
    limitsRShoulderRoll  = [-1.5620,-0.0087]
    limitsLElbowYaw      = [-2.0857, 2.0857]
    limitsRElbowYaw      = [-2.0857, 2.0857]
    limitsLElbowRoll     = [-1.5620,-0.0087]
    limitsRElbowRoll     = [ 0.0087, 1.5620]
    limitsHipPitch       = [-1.0385, 1.0385]

    pNeck =   (0.5 * (np.array(p[11]) + np.array(p[12]))).tolist()
    pMidHip = (0.5 * (np.array(p[23]) + np.array(p[24]))).tolist()

    LShoulderPitch, LShoulderRoll = keypointsToAngles.obtain_LShoulderPitchRoll_angles(pNeck, p[11], p[13], pMidHip)
    RShoulderPitch, RShoulderRoll = keypointsToAngles.obtain_RShoulderPitchRoll_angles(pNeck, p[12], p[14], pMidHip)
    
    LElbowYaw, LElbowRoll = keypointsToAngles.obtain_LElbowYawRoll_angle(pNeck, p[11], p[13], p[15])
    RElbowYaw, RElbowRoll = keypointsToAngles.obtain_RElbowYawRoll_angle(pNeck, p[12], p[14], p[16])

    HipPitch = keypointsToAngles.obtain_HipPitch_angles(pMidHip, pNeck) # This is switched, why?

    angles = [LShoulderPitch,LShoulderRoll, LElbowYaw, LElbowRoll, RShoulderPitch,RShoulderRoll, RElbowYaw, RElbowRoll, HipPitch]
    

    # print(LShoulderPitch*180/np.pi, LShoulderRoll*180/np.pi)
    angle_trace.append(angles)

    if (checkLim(LShoulderPitch, limitsLShoulderPitch) or 
        checkLim(RShoulderPitch, limitsRShoulderPitch) or
        checkLim(LShoulderRoll, limitsLShoulderRoll) or 
        checkLim(RShoulderRoll, limitsRShoulderRoll) or
        checkLim(LElbowYaw, limitsLElbowYaw) or 
        checkLim(RElbowYaw, limitsRElbowYaw) or
        checkLim(LElbowRoll, limitsLElbowRoll) or 
        checkLim(RElbowRoll, limitsRElbowRoll)):
        return False
    else:
        return True

def main():
    global angle_trace

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark
    enable_teleop = args.enable_teleop

    video = args.video
    fps = args.fps

    if len(video) == 0:
        cap = cv.VideoCapture(cap_device)
    else:
        cap = cv.VideoCapture(video)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) 

    if enable_teleop:
        # Initialize socket to send keypoints
        ss = SocketSend()
        # sr = SocketReceiveSignal()

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    if plot_world_landmark:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    while True:
        start = time()

        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)
        results_hands = hands.process(image)

        image.flags.writeable = True
        
        # Draw pose landmarks
        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                debug_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)
            # brect = calc_bounding_rect(debug_image, results.pose_landmarks)
            # debug_image = draw_landmarks(debug_image, results.pose_landmarks)
            # Rendering results
            # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        
        # Draw hands landmarks
        hand_rl = None
        if results_hands.multi_hand_landmarks:
            for num, hand in enumerate(results_hands.multi_hand_landmarks):
                mp_drawing.draw_landmarks(debug_image,
                                        hand,
                                        mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )
                # Render left or right detection
                text, coord, hand_rl = hu.get_label(num, hand, results_hands.multi_handedness, cap_width, cap_height)
                if text is not None:
                    cv.putText(debug_image, text, coord, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            
                # Draw angles to image from joint list
                if hand_rl is not None:
                    hu.draw_finger_angles_3d(debug_image, hand, hu.joint_list_low, hand_rl, cap_width, cap_height)
        
        if plot_world_landmark:
            if results.pose_world_landmarks is not None:
                plot_world_landmarks(plt, ax, results.pose_world_landmarks)

        if results.pose_world_landmarks is not None:
            cv.putText(debug_image, "TRACKING", (10, 90),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
            if not do_teleop(results.pose_world_landmarks):
                # limit reached
                cv.putText(debug_image, "LIMIT", (10, 140),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

            if enable_teleop:
                socket_stream_landmarks(ss, results.pose_world_landmarks)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 145, 255), 2, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('Pepper Teleop', debug_image)

        # if (len(video)>0):
        sleep(max(1./fps - (time() - start), 0))

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
