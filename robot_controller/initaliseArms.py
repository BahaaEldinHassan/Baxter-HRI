import rospy
# baxter_interface - Baxter Python API
import baxter_interface

from multiprocessing import Process
# initialize our ROS node, registering it with the Master
class Baxter_Arm_Control:
    def __init__(self):
        ## Initaise the interface for each arm ##
        self.limbR = baxter_interface.Limb('right')        
        self.limbL = baxter_interface.Limb('left')
        ## Get the join angels of the arms ##
        self.anglesR = self.limbR.joint_angles()
        self.anglesL = self.limbL.joint_angles()
        ## Set the end position the arms will move to ##
        self.initialR = {'right_e0': -0.002684466378799474, 'right_e1': 0.41494180312300444, 'right_s0': 0.8210632167156677, 'right_s1': 0.05944175553055978, 'right_w0': 0.04256796686382023, 'right_w1': 0.6523253300482722, 'right_w2': -0.1672039058795101}
        self.initialL = {'left_e0': -0.08935438089432535, 'left_e1': 0.5200194870931553, 'left_s0': -0.8352525390036077, 'left_s1': -0.0007669903939427069, 'left_w0': 0.18100973297047881, 'left_w1': 0.642354454927017, 'left_w2': -0.139975746894544}

    ## Move the right arm to inital position ##
    def move_right_arm(self):
        
        print(self.anglesR)
        ## If the right shoulder joint is past the table then it must lift up first to avoid hitting the table ##
        if (self.anglesR.get('right_s0') > 0.9):
            liftR = {'right_s0': 0.8, 'right_s1': -0.6, 'right_e1': 0}
            self.limbR.move_to_joint_positions(liftR)
        ## Move the arm to the intialisation point ##
        self.limbR.move_to_joint_positions(self.initialR)

    def move_left_arm(self):

        print(self.anglesL)

        ## If the left shoulder joint is past the table then it must lift up first to avoid hitting the table ##
        if (self.anglesL.get('left_s0') < -0.8):
            liftL = {'left_s0': -0.8,'left_s1': -0.6, 'left_e1': 0}
            self.limbL.move_to_joint_positions(liftL)
        ## Move the arm to the intialisation point ##
        self.limbL.move_to_joint_positions(self.initialL)

    ## Start moving the arms at the same time ##
    def move_arms(self):
        moveRightArm = Process(target=self.move_right_arm())
        moveLeftArm = Process(target=self.move_left_arm())
        moveRightArm.start()
        moveLeftArm.start()

    ## Move the arms to intial position at the same time ##
    def move_arms_test(self):
        moveRightArm = Process(target=self.limbR.move_to_joint_positions(self.initialR))
        moveLeftArm = Process(target=self.limbL.move_to_joint_positions(self.initialL))
        moveRightArm.start()
        moveLeftArm.start()

## --MAIN PROGRAM-- ##
def main():
    rospy.init_node('initalise')

    BAC = Baxter_Arm_Control()
    BAC.move_arms_test()

if __name__ == "__main__":
    main()