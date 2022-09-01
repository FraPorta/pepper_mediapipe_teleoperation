# Pepper Teleoperation using Mediapipe
## Description
This repository contains the python code to use the Softbank Robotics Humanoid Robot Pepper to approach a user and then be teleoperated by an operator using Mediapipe for human pose estimation through the webcam. From the 3D keypoints the joint angles are calculated and used as control input for the robot motors. A real-time butterworth filter is used for smoothing the control signals.

## GUI guide
[Here](https://docs.google.com/document/d/12L8lT-q1PW5xX1jOR3_xbfkzD3iiLps2_0ea9-NWRFs/edit?usp=sharing) you can find a guide for using the GUI.

## Installation guide
There are two main folders in the repository, one for the Mediapipe part in Python 3.7, one for the gui and robot control in Python 2.7.
For the first part, create a new conda environment and install requiremnts:
```bash
        conda create -n mediapipe_env python=3.7
        cd mediapipe_pose
        pip install -r requirements.txt
```

For the second part, create another conda environment and install requiremnts:
```bash
        conda create -n mediapipe_env python=2.7
        cd pepper_teleoperation
        pip install -r requirements.txt
```

Create Google Speech API credentials JSON file and copy them in a file called 'credentials.py' inside the 'pepper_teleoperation/utils' folder.


## Run the system

* Pepper real-time teleoperation after finding and approaching the user using GUI and voice controls
    * Open another terminal using `Python 2.7 32-bit` (see [miniconda2](https://repo.anaconda.com/miniconda/Miniconda2-latest-Windows-x86.exe))
    * Run `pepper_gui.py` (in `pepper_teleoperation` folder)
    * Insert the IP of your Pepper robot
    * Connect to the robot pressing the button or saying 'Connect'
    * Use the GUI buttons for approaching the user or teleoperate the robot
    * You can press the buttons also by talking (Say what is written on the button)
    * After pressing 'Start talking', the robot will repeat all the things you say, except if it recognize one of the possible commands:
        * 'move forward'
        * 'move backwards'
        * 'move right'
        * 'move left'
        * 'rotate left'
        * 'rotate right'
        * 'turn around'
        * 'stop talking'
        * 'start moving'
        * 'stop moving'
        * 'watch right'
        * 'watch left' 
        * 'watch up'
        * 'watch down'
        * 'watch ahead'
        * 'track arm'
        * 'track user'
        

## Visualize plots of the angles of a teleoperation session
* Run `plot_angles.py` (in `pepper_teleoperation` folder)
```bash
        cd ~/pepper_openpose_teleoperation/pepper_teleoperation
        python plot_angles.py --path <name of the folder where the angles are stored inside the angles_data folder> 
```
       