import time
import numpy as np
import math
from robot_model import robot_model
from scipy.spatial.transform import Rotation as R



if __name__ == "__main__":
    robot = robot_model()
    time.sleep(0.1)
    print(f"joint_current_pos = \n{robot.lcm_handler.joint_current_pos[:14]}")

    cart_pose_l = robot.Kinematic_Model.left_arm_forward_kinematics(robot.lcm_handler.joint_current_pos[:7])
    T_l = cart_pose_l.np
    position_l = T_l[:3,3]
    rotation_matrix_l = T_l[:3,:3]
    euler_zyx_l = R.from_matrix(rotation_matrix_l).as_euler('zyx', degrees=False)
    rpy_l = euler_zyx_l[::-1]
    print(f"\nleft_cart_position = {position_l}")
    print(f"\nleft_euler_angels = {rpy_l}")

    cart_pose_r = robot.Kinematic_Model.right_arm_forward_kinematics(robot.lcm_handler.joint_current_pos[7:14])
    T_r = cart_pose_r.np
    position_r = T_r[:3,3]
    rotation_matrix_r = T_r[:3,:3]
    euler_zyx_r = R.from_matrix(rotation_matrix_r).as_euler('zyx', degrees=False)
    rpy_r = euler_zyx_r[::-1]
    print(f"\nright_cart_position = {position_r}")
    print(f"\nright_euler_angels = {rpy_r}")
