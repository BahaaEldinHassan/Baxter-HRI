#!/usr/bin/env python

import rospy
import baxter_interface
from std_msgs.msg import String

def shake_hand():
    rospy.init_node('baxter_shake_hand', anonymous=True)
    limb = baxter_interface.Limb('right')

    # Set joint angles for shaking hand
    shake_angles = [
        {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.0, 'right_w0': 0.0, 'right_w1': -0.3, 'right_w2': 0.0},
        {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.5, 'right_w0': 0.0, 'right_w1': -0.5, 'right_w2': 0.0},
        {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.0, 'right_w0': 0.0, 'right_w1': -0.3, 'right_w2': 0.0},
        {'right_s0': 0.5, 'right_s1': 0.0, 'right_e0': 0.0, 'right_e1': 0.5, 'right_w0': 0.0, 'right_w1': -0.5, 'right_w2': 0.0},
    ]

    rate = rospy.Rate(1)  # 1 Hz
    for angles in shake_angles:
        limb.move_to_joint_positions(angles, timeout=2.0)
        rate.sleep()

def callback(data):
    if data.data == "shake":
        shake_hand()

def main():
    rospy.init_node('baxter_shake_hand', anonymous=True)
    shake_hand()

if __name__ == '__main__':
    main()

