#!/usr/bin/env python
#coding=utf-8

## CODE REFERENCE: https://cronfa.swan.ac.uk/Record/cronfa52455 ##
## Code was provided by supervisor, our code has been section ##
## Section header is OUR CODE followed by - Registration Number: 2309966 ##
## This was done so that assessor can see what code we produced as we are using code for external source ##
import rospy, argparse, math
from sensor_msgs.msg import Joy
import baxter_interface
from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

#　这个库也是可以控制移动的，pykdl用于描述和处理机器人模型、运动学和动力学

class MyLimb(object):
    def __init__(self, limb):
        self.limb = limb
        self.limb_interface = baxter_interface.Limb(self.limb)
        self.gripper_interface = baxter_interface.Gripper(self.limb)

        # verify robot is enabled
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
        ## Connect to a ros topic to be able to receieve the data & pass it into the callback function ##
        self.JoySub = rospy.Subscriber('/spacenav/joy', Joy, self.joint_state_joy_callback, queue_size=1)

    ## OUR CODE - Registration Number: 2309966 ##
    def joint_state_joy_callback(self, data):
        ## Get the postion & orientation of the space mouse ##
        for coords, spacenav_pose, spacenav_orient in zip(self.coordiantes, data.axes[0:4], data.axes[3:6]):
            ## Store the collected values in the arrays ##
            self.spaceNav_position[coords] = spacenav_pose
            self.spaceNav_quaternion[coords] = spacenav_orient

        ## Calculate the w value of orientation as the space mouse used only has 6 defrees of freedom ##
        if data.axes[0] == 0.0 and data.axes[1] == 0.0:
            w = self.get_pose()[6]
        else:
            ## using the x & y coordiantes to calcaulte the angle of tan ##
            theta = math.atan2(data.axes[0],data.axes[1])/math.pi*180
            ## Using cos to calcaulte the quaternions w the is missing from the space mouse ##
            w = math.cos(theta/2)
        ## Set the quaternions w ##
        self.spaceNav_quaternion['w'] = w

        ## If the user presses the left button & the gripper is open then it will close the gripper ##
        if data.buttons[0] == 1 and self.gripperOp == 0:
            ## Flips the gripper state ##
            self.gripperOp = 1
        ## If the user presses the left button & the gripper is closed then it will close the gripper ##
        elif data.buttons[0] == 1 and self.gripperOp == 1:
            ## Flips the gripper state ##
            self.gripperOp = 0
        ## Otherwise the gripper state will remain the same ##
        else:
            ## Gripper state remains unchanged ##
            self.gripperOp = self.gripperOp
    
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
            rospy.wait_for_service(node, 5.0)
            ik_response = ik_service(ik_request)
        except (rospy.ServiceException, rospy.ROSException) as error_message:
            rospy.logerr("Service request failed: %r" % (error_message,))
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
        
## --MAIN PROGRAM-- ##
def main():

    print("--wwy测试使用-- ")
    
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    args = parser.parse_args(rospy.myargv()[1:])        #　解析参数，这里没有设置，可以通过parser.add_argument(...)设定可选参数

    print("Initializing node... ")
    rospy.init_node("control_head")     #　初始化节点

    mylimb = MyLimb("right")
    rospy.on_shutdown(mylimb.clean_shutdown)        #　当关闭节点时调用mylimb.clean_shutdown函数

    while not rospy.is_shutdown():

        #　注意姿态是四元数表示，占了后面４个位置

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

        ## OUR CODE - Registration Number: 2309966 ##
        for coord, pos, poses in zip(mylimb.coordiantes, mylimb.positions, pose[0:4]):
            mylimb.positions[coord] = poses

        for coord, quat, poses in zip(mylimb.coordiantes, mylimb.quaternions, pose[4:6]):
            mylimb.quaternions[coord] = poses

        spaceNav = mylimb.moveArms()

        position = [spaceNav[0],spaceNav[1],spaceNav[2],pose[3],pose[4],pose[5], pose[6]]

        baxterMove = mylimb.baxter_ik_move(position)
        if baxterMove == None:
            continue
        else:
            print("BaxterMove", baxterMove)
            mylimb.limb_interface.move_to_joint_positions(baxterMove, timeout=0.07)
        
        if mylimb.gripperOp == 0:
            print("Open Gripper")
            mylimb.gripper_interface.open(timeout=5)
        else:
            print("Close Gripper")
            mylimb.gripper_interface.close(timeout=5)
        ## END OF OUR CODE - Registration Number: 2309966 ##
        print('\r')

if __name__ == '__main__':
    main()
    
    #　限位测算：
    #　笛卡尔空间：ｘ[0.5, 0.8]，y[0, 0.45]，ｚ[-0.13, 0.56]，注意四元数的顺序是xyzw，实轴是w，正常向下大约是[0, 0.99, 0.001, 0.001]
