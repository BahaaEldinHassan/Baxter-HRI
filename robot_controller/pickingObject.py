#!/usr/bin/env python
#coding=utf-8

## CODE REFERENCE: https://cronfa.swan.ac.uk/Record/cronfa52455 ##
## Code was provided by supervisor, our code has been section ##
## Section header is OUR CODE followed by - Registration Number: 2309966 ##
## This was done so that assessor can see what code we produced as we are using code for external source ##
import rospy, argparse, math, os, time, math
import numpy as np
import baxter_interface
from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

from DMP_quaternion.dmp_position import PositionDMP
from DMP_quaternion.dmp_rotation import RotationDMP
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

#　这个库也是可以控制移动的，pykdl用于描述和处理机器人模型、运动学和动力学

filepath = os.path.dirname(os.path.abspath(__file__))

class MyLimb(object):
    def __init__(self, limb):
        self.limb = limb
        self.limb_interface = baxter_interface.Limb(self.limb)
        self.gripper_interface = baxter_interface.Gripper("right")

        # verify robot is enablede boot was on. I just turned it off and it worked for me. Try to
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        ## Set the gripper state to be open when starting ##
        self.gripperOp = 0
        ## Initalise the position & orientation of the robot to zero ##
        self.coordiantes = ['x', 'y', 'z']
        self.quaternion = ['x', 'y', 'z', 'w']
        self.positions = {coord: 0.0 for coord in self.coordiantes}
        self.quaternions = {coord: 0.0 for coord in self.quaternion}
        ## Initalise the space mouse position & orientation of the robot to zero ##
        self.spaceNav_position = {coord: 0.0 for coord in self.coordiantes}
        self.spaceNav_quaternion = {coord: 0.0 for coord in self.quaternion}
        print("Running. Ctrl-c to quit")

    ## OUR CODE - Registration Number: 2309966 ##
    def moveArms(self):
        ## If the robots arm goes below a certain z coordiate value then the arm will move away from the table ##
        if self.positions['z'] < -0.224:
            ## Warning user that the arm is too close to the table ##
            print("Arm too close to table! Moving arm away to avoid collision")
            self.positions['x'] = self.positions['x'] 
            self.positions['y'] = self.positions['y']
            ## Set z coordiante to above the table to avoid collison ##
            self.positions['z'] = -0.2
        ## Otherwise the robot keeps moving ##
        else:
            ## If the space mouse is pushed forward the the arm will move forward ##
            if self.spaceNav_position['x'] > 0.5:
                self.positions['x'] = self.get_pose()[0] + 0.1
            ## If the space mouse is pushed backwards the the arm will move forward ##
            elif self.spaceNav_position['x'] < -0.5:
                self.positions['x'] = self.get_pose()[0] - 0.1
            ## If the space mouse is pushed forward the the arm will orientate correctly ##
            elif self.spaceNav_quaternion['x'] > 0.5:
                self.quaternions['x'] = self.get_pose()[5] + 0.1
            ## If the space mouse is pushed forward the the arm will orientate correctly ##
            elif self.spaceNav_quaternion['x'] < -0.5:
                self.quaternions['x'] = self.get_pose()[5] - 0.1

            ## If the arm that is currently selected to move is left then the y coordiante mirror the arm ##
            if self.limb == "left":
                ## If the space mouse is pushed right the the arm will move right ##
                if self.spaceNav_position['y'] > 0.5:
                    self.positions['y'] = self.get_pose()[1] + 0.1
                ## If the space mouse is pushed left the the arm will move left ##
                elif self.spaceNav_position['y'] < -0.5:
                    self.positions['y'] = self.get_pose()[1] - 0.1
                ## If the space mouse is not being used then the arm remains where it is ##
                else:
                    self.positions['y'] = self.get_pose()[2]
            ## If the are that is currently selected to move is right then the y coordiante mirror the arm ##   
            elif self.limb == "right":
                ## If the space mouse is pushed left the the arm will move left ##
                if self.spaceNav_position['y'] > 0.5:
                    self.positions['y'] = self.get_pose()[1] - 0.1
                ## If the space mouse is pushed right the the arm will move right ##
                elif self.spaceNav_position['y'] < -0.5:
                    self.positions['y'] = self.get_pose()[1] + 0.1
                ## If the space mouse is not being used then the arm remains where it is ##
                else:
                    self.positions['y'] = self.get_pose()[2]

            ## If the space mouse is pushed left the the arm will orientate correctly ##
            if self.spaceNav_quaternion['y'] > 0.5:
                self.quaternions['y'] = self.get_pose()[5] + 0.1
            ## If the space mouse is pushed right the the arm will orientate correctly ##
            elif self.spaceNav_quaternion['y'] < -0.5:
                self.quaternions['y'] = self.get_pose()[5] - 0.1
            ## If the space mouse is pulled up the the arm will move up ##
            elif self.spaceNav_position['z'] > 0.5:
                self.positions['z'] = self.get_pose()[2] + 0.1
            ## If the space mouse is pushed down the the arm will move down ##
            elif self.spaceNav_position['z'] < -0.5:
                self.positions['z'] = self.get_pose()[2] - 0.1
            ## If the space mouse is pulled up the the arm will orientate correctly ##
            elif self.spaceNav_quaternion['z'] > 0.5:
                self.quaternions['z'] = self.get_pose()[5] + 0.1
            ## If the space mouse is pushed down the the arm will orientate correctly ##
            elif self.spaceNav_quaternion['z'] < -0.5:
                self.quaternions['z'] = self.get_pose()[5] - 0.1
            ## If the space mouse is not being used then the arm remains where it is ##
            else:
                self.positions['z'] = self.get_pose()[2]
                self.quaternions['z'] = self.get_pose()[5]
        
        ## Calculate the w value of orientation as the space mouse used only has 6 defrees of freedom ##
        if self.spaceNav_position['x'] == 0.0 and self.spaceNav_position['y'] == 0.0:
            w = self.get_pose()[6]
        else:
            ## using the x & y coordiantes to calcaulte the angle of tan ##
            theta = math.atan2(self.spaceNav_position['x'],self.spaceNav_position['y'])/math.pi*180
            ## Using cos to calcaulte the quaternions w the is missing from the space mouse ##
            w = math.cos(theta/2)
        ## Set the quaternions w ##
        self.positions['w'] = w

        ## Returns the position & orientation the robot should be based on the space mouse input ##
        return (self.positions['x'],self.positions['y'],self.positions['z'], self.quaternions['x'], self.quaternions['y'], self.quaternions['z'], self.quaternions['w'])
    ## END OF OUR CODE - Registration Number: 2309966 ##

    # get the pose
    def get_pose(self):
        quaternion_pose = self.limb_interface.endpoint_pose()
        position = quaternion_pose['position']
        quaternion = quaternion_pose['orientation']

        return (position[0],position[1],position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        
    # get the joint
    def get_joint(self):
        joint= self.limb_interface.joint_angles()

        return joint

    # move a limb，调用ik服务，返回逆解
    def baxter_ik_move(self, quaternion_pose):

        node = "ExternalTools/" + self.limb + "/PositionKinematicsNode/IKService"
        ik_service = rospy.ServiceProxy(node, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id="base")
       
        pose_stamp = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=quaternion_pose[0],
                    y=quaternion_pose[1],
                    z=quaternion_pose[2],
                ),
                orientation=Quaternion(
                    x=quaternion_pose[3],
                    y=quaternion_pose[4],
                    z=quaternion_pose[5],
                    w=quaternion_pose[6],
                ),
            ),
        )

        ik_request.pose_stamp.append(pose_stamp)
        try:
            # rospy.wait_for_service(node, 5.0)
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException) as error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
            # sys.exit("ERROR - baxter_ik_move - Failed to append pose")
            print("ERROR - baxter_ik_move - Failed to append pose")
            return
        
        if ik_response.isValid[0]:
            print("PASS: Valid joint configuration found")
            # convert response to joint position control dictionary
            limb_joints = dict(zip(ik_response.joints[0].name, ik_response.joints[0].position))
            return limb_joints
           
        else:
            # little point in continuing so exit with error message
            print("requested move =", quaternion_pose)
            print("ERROR - baxter_ik_move - Failed to append pose")
            return 

    def clean_shutdown(self):
        """
        Exits example cleanly by moving head to neutral position and
        maintaining start state
        """
        print("\nExiting example...")
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()
    def gripperclose(self):
        ###################INIT#########################################
	    #rospy.init_node('Hello_Baxter')
        limbr = baxter_interface.Limb('right')
        limbl = baxter_interface.Limb('left')
        gripr=baxter_interface.Gripper('right')
        gripl=baxter_interface.Gripper('left')
	    #gripr.calibrate()
	    #gripl.calibrate()

	    #limbl.set_joint_position_speed(0.1)

	    #limbl.move_to_neutral()
        re = gripl.close()
        print(re)
    def gripperopen(self):
        ###################INIT#########################################
        
        limbr = baxter_interface.Limb('right')
        limbl = baxter_interface.Limb('left')
        gripr=baxter_interface.Gripper('right')
        gripl=baxter_interface.Gripper('left')
	    #gripr.calibrate()
	    #gripl.calibrate()

	    #limbl.set_joint_position_speed(0.1)

	    #limbl.move_to_neutral()
        re=gripl.open()
        print(re)

## The dynamic movement primitives function to calcualte the trajectory of the robot arms ##
def DMP(startPos, goalPos, filename):
    ## SOURCE CODE REFERENCE: https://github.com/mathiasesn/obstacle_avoidance_with_dmps/tree/dmp_online_obs_avoid/DMP ##
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt(filepath+"/DMP_quaternion/"+filename, delimiter=" ", skiprows=1)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]

    demo_o = demo[:,3:demo.shape[-1]-1]

    # Check for sign flips
    for i in range(len(demo_o)-1):
        if demo_o[i].dot(demo_o[i+1]) < 0:
            demo_o[i+1] *= -1

    theta = [np.linalg.norm(v) for v in demo_o]
    axis = [v/np.linalg.norm(v) for v in demo_o]

    demo_q = np.array([Quaternion(axis=a,radians=t) for (a,t) in zip(axis,theta)])

    for i in range(len(demo_q)-1):
        if np.array([demo_q[i][0], demo_q[i][1], demo_q[i][2], demo_q[i][3]]).dot(np.array([demo_q[i+1][0], demo_q[i+1][1], demo_q[i+1][2], demo_q[i+1][3]])) < 0:
            demo_q[i+1] *= -1

    demo_quat_array = np.empty((len(demo_q),4))
    for n, d in enumerate(demo_q):
        demo_quat_array[n] = [d[0],d[1],d[2],d[3]]

    # TODO: In both canonical_system.py and dmp_position.py you will find some lines missing implementation.
    # Fix those first.

    N = 100  # TODO: Try changing the number of basis functions to see how it affects the output.
    dmp = PositionDMP(n_bfs=N, alpha=48.0)
    dmp.train(demo_p, t, tau)

    # Rotation...
    dmp_rotation = RotationDMP(n_bfs=N, alpha=48.0)
    dmp_rotation.train(demo_q, t, tau)

    # a different starting point for the dmp:
    if startPos != demo[0]:
        dmp.p0 = np.array([startPos[0], startPos[1], startPos[2]])

    # a different goal point:
    if goalPos != demo[-1]:
        print(goalPos[0])
        dmp.gp = np.array([goalPos[0], goalPos[1], goalPos[2]])

    # a different time constant:
    tau = 0.001 * len(demo)

    # Generate an output trajectory from the trained DMP
    dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)

    dmp_r, dmp_dr, dmp_ddr = dmp_rotation.rollout(t, tau)
    result_quat_array = np.empty((len(dmp_r),4))
    for n, d in enumerate(dmp_r):
        result_quat_array[n] = [d[0],d[1],d[2],d[3]]

    ## END OF SOURCE CODE REFERENCE ##

    return dmp_p, dmp_dp, dmp_ddp, dmp_r, dmp_dr, dmp_ddr

def main():
    ## Gets which limb will be used to pick up & place the object ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limb')  #　解析参数，这里没有设置，可以通过parser.add_argument(...)设定可选参数
    args = parser.parse_args()

    print("Initializing node... ")
    rospy.init_node("control_head")     #　初始化节点

    ## Initalise the selected robot arm ##
    mylimb = MyLimb(args.limb)
    rospy.on_shutdown(mylimb.clean_shutdown)        #　当关闭节点时调用mylimb.clean_shutdown函数
    ## If the robot arm selected is right then the trajectories for the arm are loaded ##
    if args.limb == "right":
        filenames = ["trajectory_1.dat", "trajectory_2.dat", "trajectory_4.dat"]
    ## If the robot arm selected is left then the trajectories for the arm are loaded ##
    else:
        filenames = ["trajectory_3.dat", "trajectory_5.dat", "trajectory_6.dat"]

    ## Set the first trajectory to be selected ##
    f = 0
    ## Set that the gripper state to be none ##
    gripperAction = None

    ## While the robot is not shutdown then the loop will keep running ##
    while not rospy.is_shutdown():
        ## Load the selected trajectory to be used ##
        demo = np.loadtxt(filepath+"/DMP_quaternion/"+filenames[f], delimiter=" ") #　注意姿态是四元数表示，占了后面４个位置

        ## Get the current pose of the robot selected arm ##
        pose = mylimb.get_pose()
        ## Get the joints of the robot selected arm ##
        joint = mylimb.get_joint()
        ## Get the current joint velocity of the robot selected arm ##
        joint_vel = mylimb.limb_interface.joint_velocities()
        ## Get the current joint effor of the robot selected arm ##
        joint_effort = mylimb.limb_interface.joint_efforts()
        ## Display current position of selected arm ##
        print(pose)
        ## Display selected trajectory of selected arm ##
        print(filenames[f])

        ## OUR CODE - Registration Number: 2309966 ##
        ## Set the start position for the DMP to follow ##
        startPos = [pose[0], pose[1], pose[2]]
        ## Set the end goal based on the selected trajectory ##
        if filenames[f] == "trajectory_1.dat":
            ## As trajectory_1 is after reaching the object the gripper need to close to pick the object up ##
            ## Gripper state is changed to closed ##
            gripperAction = "close"
            goalPos = [0.7695371895035115, -0.5102693312427654, -0.21696345239128245]
        elif filenames[f] == "trajectory_4.dat":
            goalPos = [0.8591636099865624, -0.09691726451738228, -0.23201175361031817]
        elif filenames[f] == "trajectory_2.dat":
            ## As trajectory_2 is right after the object is placed the gripper need to open first ##
            ## Gripper state is changed to open ##
            gripperAction = "open"
            goalPos = [0.6743577591275197, -0.8490598637769811, -0.1042401166567425]
        elif filenames[f] == "trajectory_3.dat" or filenames[f] == "trajectory_5.dat":
            goalPos = [-0.7045878428750021, -0.4736438281833637, 0.21864764518107155]
            print("GOALPOSE", goalPos)
        else:
            goalPos = [demo[-1][0], demo[-1][1], demo[-1][2]]

        time.sleep(0.8)
        
        ## The trajectory is calcualted with the new start & end goal ##
        p, dp, ddp, r, dr, ddr = DMP(startPos, goalPos, filenames[f])

        ## Displays the gripper state ##
        print(gripperAction)

        ## Check if the gripper state should be close ##
        if gripperAction == "close":
            ## Closes the gripper of the selected arm ##
            mylimb.gripper_interface.close(timeout=0.05)
            time.sleep(0.8)
            ## Resets the gripper state to none so no action should be performed on the gripper ##
            gripperAction == None
        elif gripperAction == "open":
            ## opens the gripper of the selected arm ##
            mylimb.gripper_interface.open(timeout=0.05)
            time.sleep(0.8)
            ## Resets the gripper state to none so no action should be performed on the gripper ##
            gripperAction == None

        ## Itertates through all the calculated positions of the newly trained trajectory ##
        for i, j in zip(p, r):
            ## If the robot is shut down or the arm has reached it final position the the loop is broken ##
            if rospy.is_shutdown() or (filenames[f] == "trajectory_1.dat" and i[1] <= goalPos[1] and i[2] <= goalPos[2]):
                break
            ## As the orientation vlaues of Quaternion are (w, x, y, z) numpy roll is used to make them (x, y, z, w)
            ort = np.roll(j.elements, -1)
            ## Get the position & orientation of the trained trajectory ##
            pos = np.append(i, [pose[3], pose[4], pose[5], pose[6]], axis=0)
            ## Display the trajectory position ##
            print(pos)
            ## Calcualte the inverse kinematics in a cartisian space ##
            baxterMove = mylimb.baxter_ik_move(pos)
            ## If invalid position from the IKSolver is found then keep looping till a valid one is found ##
            if baxterMove == None:
                continue
            else:
                ## Move the arm to the points from the calculated trajectory ##
                mylimb.limb_interface.move_to_joint_positions(baxterMove, timeout=0.05)

        ## Get the next trajectory for the arm to follow & DMP to train with ##
        f += 1
        ## If all trajectory files have been read then end the picking & placing object action ##
        if f == len(filenames):
            break
        ## END OF OUR CODE - Registration Number: 2309966 ##
        print('\r')
        
        
## --MAIN PROGRAM-- ##        
if __name__ == '__main__':
    main()
    
    #　限位测算：
    #　笛卡尔空间：ｘ[0.5, 0.8]，y[0, 0.45]，ｚ[-0.13, 0.56]，注意四元数的顺序是xyzw，实轴是w，正常向下大约是[0, 0.99, 0.001, 0.001]