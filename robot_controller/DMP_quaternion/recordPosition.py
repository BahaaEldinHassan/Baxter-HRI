import rospy, baxter_interface
import time as t
import numpy as np
import os
from baxter_interface import CHECK_VERSION
from sensor_msgs.msg import JointState

## Class that record the trajectory of the robot arm ##
class MyLimb(object):
    def __init__(self, limb):
        ## Initalise the interface or the selected limb ##
        self.limb = limb
        self.limb_interface = baxter_interface.Limb(self.limb)
        ## Connect to a ros topic to be able to receieve the data & pass it into the callback function ##
        self.js = rospy.Subscriber("/robot/joint_states",  JointState, self.joint_state_callback)
        ## Initalise arm velocity as nothing ##
        self.velocity = []
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    ## Function to get the position & orientation of the selected arm ##
    def get_positions(self):
        ## Gets the position & orientation of the selected ar
        quaternion_position = self.limb_interface.endpoint_pose()
        position = quaternion_position['position']
        quaternion = quaternion_position['orientation']

        ## Returns the position & orientation of the selected arm ##
        return (position[0],position[1],position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    
    ## Stores the current velocity of the selected robot arm ##
    def joint_state_callback(self, data):
        self.velocity = data.velocity

    ## Makes sure the robot shutsdown cleanly ##
    def clean_shutdown(self):
        """
        Exits example cleanly by moving head to neutral position and
        maintaining start state
        """
        print("\nExiting example...")
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

## --MAIN PROGRAM-- ##
if __name__ == '__main__':
    rospy.init_node("control_head")
    mylimb = MyLimb("right")
    startPos = mylimb.get_positions()
    Pos = [list(startPos)]
    moving = False
    jointNames = mylimb.limb_interface.joint_names()
    rospy.on_shutdown(mylimb.clean_shutdown)
    v = mylimb.limb_interface.joint_velocities()
    time = 0

    ## While the robot has not been shutdown & that about five seconds has not passed since starting the loop ##
    while not rospy.is_shutdown() and time < 300:
        ## Gets & stores the currnet position of the selected robot arm ##
        currentPos = mylimb.get_positions()
        t.sleep(0.01)
        ## Gets & stores the current velocity of the selected arm ##
        v = mylimb.limb_interface.joint_velocities()
        print(v)
        ## Check if any of the velocities within the array are greater than 0.1 or less than -0.1 ##
        moving = any((elms > 0.1 or elms < -0.1) for elms in [v[name] for name in jointNames])
        
        ## If any of the values checked are above or below the threshold then the are is detected as moving ##
        if (moving):
          print("Arm is moving")
          print(currentPos)
          ## Append the current position of the selected arm to be used later ##
          Pos.append(list(currentPos))
          ## Reset the timer of how long the arm has been idle for ##
          time = 0

        ## If the selected arm is detected as not moving then increase the idle timer by 1 ##
        else:
          time += 1
          print("Arm is not moving")

    ## Get the file path of this file ##
    filenames = []
    files = os.listdir(".")
    ## Sort the file names in alphabetical order ##
    files.sort()
    ## For each file that is found ##
    for filename in files:
        ## Select only the files begining with trajectory_ & have a file extension of .dat ##
        if filename.startswith("trajectory_") and filename.endswith(".dat"):
            ## Store those file names ##
            filenames.append(filename)

    ## Get the last found file of the trajectory files & increment the number by 1 ##
    trajectoryFilename = str(int(str(filenames[-1]).split(".")[0].split("_")[1])+1)
    ## Print what the file name will be for the new trajectory to be stored in ##
    print("\nTrajectory was saved to the file trajectory_"+trajectory+'.dat')
    ## Save the recored points of the trajectory in a text file with the extension .dat ##
    np.savetxt("trajectory_"+trajectoryFilename+'.dat', Pos)