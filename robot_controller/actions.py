import rospy, random, argparse, subprocess, shaking_hands
import time as t
import baxter_interface
from baxter_interface import CHECK_VERSION

## Class that contains all the response actions the robot can perfrom ##
class Actions(object):
    def __init__(self):
        ## Intialises the interfaces for head & arms ##
        self.head = baxter_interface.Head()
        self.limbR = baxter_interface.Limb('right')
        self.limbL = baxter_interface.Limb('left')
        ## Sets the default position for both arms of baxter ##
        self.returnState_l = {'left_e0': -0.08935438089432535, 'left_e1': 0.5200194870931553, 'left_s0': -0.8352525390036077, 'left_s1': -0.0007669903939427069, 'left_w0': 0.18100973297047881, 'left_w1': 0.642354454927017, 'left_w2': -0.139975746894544}
        self.returnState_r = {'right_e0': -0.002684466378799474, 'right_e1': 0.41494180312300444, 'right_s0': 0.8210632167156677, 'right_s1': 0.05944175553055978, 'right_w0': 0.04256796686382023, 'right_w1': 0.6523253300482722, 'right_w2': -0.1672039058795101}

    ## Shakes the robots head back & forth twince ##
    def shakingHead(self):
        ## Sets the robots head to face forward ##
        self.head.set_pan(0.0)
        t.sleep(2)
        ## Move the robot's head left to right ##
        self.head.set_pan(-0.4, 0.3)
        self.head.set_pan(0.4, 0.3)
        self.head.set_pan(0.0)
    
    ## Nods the robots head twice ##
    def noddingHead(self):
        ## Sets the robots head to face forward ##
        self.head.set_pan(0.0)
        t.sleep(2)
        ## Performs the head nodding command twice giving the action for agreeing ##
        for i in range(2):
            self.head.command_nod()

    ## Runs the robot picking & places action ##
    def pickingObject(self):
        ## Starts the node that contains the DMP tree for the picking & placing objects ##
        subprocess.call(["python3", "pickingObject.py", "--limb", "right"])

    ## Waves both arms of the robot one after the other ##
    def waving(self):
        ## Set the positions of the arms for waving ##
        wave_1_l = {'left_e0': -1.2547962844902685, 'left_e1': 1.3326458094754532, 'left_s0': 0.2247281854252131, 'left_s1': -0.7075486384121471, 'left_w0': 0.858262250821889, 'left_w1': -1.5715633171886063, 'left_w2': -0.1802427425765361}
        wave_1_r = {'right_e0': 1.0484758685196802, 'right_e1': 1.64059245264345, 'right_s0': -0.20708740636453085, 'right_s1': -0.9621894492011258, 'right_w0': -0.6657476619422695, 'right_w1': -1.5715633171886063, 'right_w2': -0.24045148850103862}
        wave_2_l = {'left_e0': -1.2547962844902685, 'left_e1': 1.3326458094754532, 'left_s0': 0.2247281854252131, 'left_s1': -0.7075486384121471, 'left_w0': 1.3, 'left_w1': -1.5715633171886063, 'left_w2': -0.1802427425765361}
        wave_2_r = {'right_e0': 1.0484758685196802, 'right_e1': 1.64059245264345, 'right_s0': -0.20708740636453085, 'right_s1': -0.9621894492011258, 'right_w0': -1.3, 'right_w1': -1.5715633171886063, 'right_w2': -0.24045148850103862}
        wave_3_l = {'left_e0': -1.2547962844902685, 'left_e1': 1.3326458094754532, 'left_s0': 0.2247281854252131, 'left_s1': -0.7075486384121471, 'left_w0': 0.4, 'left_w1': -1.5715633171886063, 'left_w2': -0.1802427425765361}
        wave_3_r = {'right_e0': 1.0484758685196802, 'right_e1': 1.64059245264345, 'right_s0': -0.20708740636453085, 'right_s1': -0.9621894492011258, 'right_w0': 0.4, 'right_w1': -1.5715633171886063, 'right_w2': -0.24045148850103862}

        ## Move the arm up into a waving position ##
        self.limbL.move_to_joint_positions(wave_1_l, timeout=2.0)

        ## Rotates the wrist (left_w0) back & forth twice giving the action of waving ##
        for i in range(2):
            self.limbL.move_to_joint_positions(wave_2_l, timeout=0.85)
            self.limbL.move_to_joint_positions(wave_3_l, timeout=0.85)

        ## Moves the arm back to the center point of the wave, this sets the end of the wave action ##
        self.limbL.move_to_joint_positions(wave_1_l, timeout=2.0)
        ## Moves the arm back to the default position ready to perfrom the next action that will be recognised ##
        self.limbL.move_to_joint_positions(self.returnState_l, timeout=10.0)
        
        ## Move the arm up into a waving position ##
        self.limbR.move_to_joint_positions(wave_1_r, timeout=5.0)

        ## Rotates the wrist (right_w0) back & forth twice giving the action of waving ##
        for i in range(2):
            self.limbR.move_to_joint_positions(wave_2_r, timeout=0.85)
            self.limbR.move_to_joint_positions(wave_3_r, timeout=0.85)

        ## Moves the arm back to the center point of the wave, this sets the end of the wave action ##
        self.limbR.move_to_joint_positions(wave_1_r, timeout=2.0)
        ## Moves the arm back to the default position ready to perfrom the next action that will be recognised ##
        self.limbR.move_to_joint_positions(self.returnState_r, timeout=10.0)

    # Shake the hand of the user
    def shakingHand(self):
        # Set joint angles for shaking hand
        shake_angles = [
            {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.0, 'right_w0': 0.0, 'right_w1': -0.3, 'right_w2': 0.0},
            {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.5, 'right_w0': 0.0, 'right_w1': -0.5, 'right_w2': 0.0},
            {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.0, 'right_w0': 0.0, 'right_w1': -0.3, 'right_w2': 0.0},
            {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.5, 'right_w0': 0.0, 'right_w1': -0.5, 'right_w2': 0.0},
        ]

        rate = rospy.Rate(1)  # 1 Hz
        # Performs all the number of movements in the shake angles array
        for angles in shake_angles:
            self.limbR.move_to_joint_positions(angles, timeout=2.0)
            rate.sleep()

## --MAIN PROGRAM-- ##
def main():
    ## Gets the argument action to determin which action to perform ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action')      
    args = parser.parse_args()

    rospy.init_node("actions")
    print("Initializing node... ")
    ## Intalisies the action class ##
    actions = Actions()
    ## If the action detected is 0 then shaking head action is performed in response ##
    if args.action == "0":
        print("shaking head... ")
        actions.shakingHead()
        print("Done.")
    ## If the action detected is 1 then nod head action is performed in response ##
    elif args.action == "1":
        print("Nodding head... ")
        actions.noddingHead()
        print("Done.")
    ## If the action detected is 2 then picking & placing object action is performed in response ##
    elif args.action == "2":
        print("Picking & placing object...")
        actions.pickingObject()
        print("Done")
    ## If the action detected is 3 then waving action is performed in response ##
    elif args.action == "3":
        print("Waving...")
        actions.waving()
        print("Done")
    ## If the action detected is 4 then shaking hand action is performed in response ##
    elif args.action == "4":
        print("Shaking hand")
        actions.shakingHand()
        print("Done")
    ## If no action detected then print no action found ##
    else:
        print("Action not found!")

if __name__ == "__main__":
    main()