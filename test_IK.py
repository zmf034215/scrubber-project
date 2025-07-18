import time
import numpy as np
import math
from math import sin, cos
from robot_model import robot_model
import pinocchio
import sys

def random_cart_pos(current_cart_pos):
    """
    输入当前的位姿，进行随机的改变，用于迭代数值逆解正确性的检验
    """
    trans = current_cart_pos.translation
    rot = current_cart_pos.rotation
    incre_trans = np.random.rand(3) * 0.1
    print(incre_trans)   
    
    incre_rot = np.random.rand(3) * 0.1 - 0.05 # 分别代表旋转角度、x、y、z
    x,y,z = incre_rot
    print(x,y,z)
    Rx = np.array([
        [1, 0, 0],
        [0, cos(x), -sin(x)],
        [0, sin(x), cos(x)]
    ])
    Ry = np.array([
        [cos(y), 0, sin(y)],
        [0, 1, 0],
        [-sin(y), 0, cos(y)]
    ])
    Rz = np.array([
        [cos(z), -sin(z), 0],
        [sin(z), cos(z), 0],
        [0, 0, 1]
    ])
    target_rot = Rx @ Ry @ Rz @ rot
    target_trans = trans + incre_trans
    return target_trans, target_rot

if __name__ == '__main__':
    """
    检测逆解函数正确性
    """
    robot = robot_model()
    time.sleep(1)
    if len(sys.argv) > 1:
        np.random.seed(int(sys.argv[1]))
    else:
        np.random.seed(int(time.time()))
    # 先计算当前关节角下的末端位置（默认情况下，无设置无采集数据时为全0数组）
    current_cart_pos = robot.Kinematic_Model.left_arm_forward_kinematics(robot.lcm_handler.joint_current_pos[:7])
    print("当前末端位置为：", current_cart_pos)
    # 对当前位置加上一个小的偏移量
    target_trans, target_rot = random_cart_pos(current_cart_pos)
    target_pose = pinocchio.SE3(np.array(target_rot), np.array(target_trans))
    print("随机的笛卡尔空间位姿为：","\n")
    print("位置：", target_pose, "\n")


    # 计算逆解结果
    robot.Kinematic_Model.left_arm_inverse_kinematics(target_rot, target_trans, robot.lcm_handler.joint_current_pos[:7])
    left_arm_joint_position = robot.Kinematic_Model.left_arm_interpolation_result
    print("初始笛卡尔空间误差", np.linalg.norm(target_pose.translation - current_cart_pos.translation), np.linalg.norm(target_pose.rotation - current_cart_pos.rotation),"\n")

    current_cart_pos = robot.Kinematic_Model.left_arm_forward_kinematics(left_arm_joint_position)
    print("正运动学解后末端位置为：", current_cart_pos)


    if robot.Kinematic_Model.left_arm_inverse_kinematics_solution_success_flag:
        print("逆解成功，逆解结果为：", left_arm_joint_position,"\n")
        print("逆解结果与实际结果误差为：", np.linalg.norm(target_pose.translation - current_cart_pos.translation), np.linalg.norm(target_pose.rotation - current_cart_pos.rotation),"\n")
    else:
        print("逆解结果与实际结果误差为：", np.linalg.norm(target_pose.translation - current_cart_pos.translation), np.linalg.norm(target_pose.rotation - current_cart_pos.rotation),"\n")



    