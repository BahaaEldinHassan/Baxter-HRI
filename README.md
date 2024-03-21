# Robot Human Interaction

## System Requirements
To be able to use this project the following must be installed on your local system:

  - `Ubuntu 20.04.06 LTS`
  - `Python 3.8.10 or 3.11+`
  - `ROS Noetic`

These are a must to be able to use this project as there are what the project was developed on & has not been tested on any other version of the listed above.

## Computer Vision

To run the vision modules, we need `Python 3.11+` (with `pip`). The other required Python packages are described in `requirements.txt`. Before executing any of the modules described in this section, make sure to be using this Python version and have the requirements installed.

Note: You can install the required Python packages with this command (run this from the root of the project directory):

```shell
pip install -r requirements.txt
```

Then change the directory to `ml_nn_vision`:

```shell
cd ml_nn_vision
```

We have two modules for the vision part. Their functionalities and how to run them are described below:

### Savant

This module includes the CNN classification model used to train the model. It also preprocesses the dataset and takes care of data padding (e.g., the hand and body gestures have different dimensionality, to match it to each other this module pads the lower dimensional one with some constant values).

To run the module, use the following command:

```shell
python -m savant -e <epoch_count> -lr <learning_rate> -cp <checkpoint_filename>
```

All the available CLI options are as follows:

```shell
Usage: python -m savant [OPTIONS]

Options:
  -e, --num-epochs INTEGER    Number of epochs.
  -lr, --learning-rate FLOAT  Learning rate.
  -d, --dataset [combined]    Dataset to use.
  -M, --model [cnn]           Neural network model to use.
  -m, --momentum FLOAT        SGD Momentum.
  -g, --num-gates INTEGER     Number of gates to use (for GRU).
  -cp, --checkpoint TEXT      Checkpoint file to use.
  --help                      Show this message and exit.
```

### Wisdom

This module is used to take care of the recording and collection of test data, use trained model with multiple camera feeds (Intel RealSense, laptop camera module etc) to infer in real time, depth testing with YOLOv8 object detection.

#### Recording and Training Data Collection

To collect the data, you can use this command:

```shell
python -m wisdom record -l thumbs_up -cf camera -r hand
```

All the available commands are as follows:

```shell
Usage: python -m wisdom record [OPTIONS]

Options:
  -l, --label [thinking|victory|beckoning|closed_fist|handshake|open_palm|thumbs_down|thumbs_up]
                                  Label to use.  [required]
  -cf, --camera-feed [camera|realsense]
                                  Camera feed to use.  [required]
  -r, --recorder [body|hand|yolo]
                                  Recorder to use.  [required]
  --help                          Show this message and exit.
```

#### RealTime Inference

To run real-time inference, you can use this command:

```shell
python -m wisdom infer -cp <checkpoint_filename>
```

Note: Make sure you have trained the model and already have a model checkpoint file.

All the available commands are as follows:

```shell
Usage: python -m wisdom infer [OPTIONS]

Options:
  -cf, --camera-feed [camera|realsense]
                                  Camera feed to use.  [required]
  -cp, --checkpoint TEXT          Checkpoint file to use.  [required]
  --help                          Show this message and exit.
```

#### RealTime Depth Inference with YOLOv8

To run the real-time depth inference using the Intel RealSense camera with YOLO, you can use this command:

```shell
python -m wisdom record -l thumbs_up -cf realsense -r yolo
```

## Robot Controller

#### Requiements
For the robot controller ensure that `ROS Noetic` has been installed & the local operating system is `Ubuntu 20.04.6` as per the system requirements section states.

When downloading the project ensure that it is placed within the ros workspace & inside the source folder otherwise the code will not be able to connec to baxter robot.

spacenav_node must be install to be able to use the manual controll part of the project, to install the following commands must be ran:
```shell
$ sudo apt install spacenavd
$ sudo apt install ros-noetic-spacenav-node
```

#### Setting up & running the robot controller
First the user must open a terminal & cd into the ros workspace
```shell
$ cd ros_ws 
```

Then the user must run the sh file to connect to baxter
```shell
$ ./baxter.sh
```

To get the spacenav_node topics to use the 3D connection space mouse, a command line must be run first
```shell
$ roslaunch spacenav_node classic.launch
```

Open a new terminal if the user is not already inside the ros workspace when opening a new terminal then the user must run the first command again. Once inside the ros_ws re-run the sh command again. Once the user has connected to the baxter again the user must enable the robot using
```shell
$ rosrun baxter_tool enable_robot.py -e 
```

Once the robot has been enabled the user can now cd into the robot controller workspace (the [path_to_robot_controller] should be replace with your path to the robot_controller folder)
```shell
$ cd src/[path_to_robot_controller]/robot_controller 
```

Everything should now be set up for the user to start running the main program using the command
```shell
$ python3 MainMenu.py 
```
