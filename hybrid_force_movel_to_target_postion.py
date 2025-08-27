import time
import numpy as np
import math
from robot_model import robot_model



if __name__ == "__main__":
    robot = robot_model()
    robot.Force_Control.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)
    # 该参数设置为0 是不进行碰撞检测
    robot.Collision_Detection.collision_detection_level = 0
    hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
    hand_home_pos = list(hand_home_pos / 180 * np.pi)

    start_pose = np.array([0.3, -0.55, 0.2, -1.8, 0.75, 1.8])
    robot.Kinematic_Model.move_to_start_pose('right', start_pose)



    # 只对左臂在xy平面进行运动
    current_joint = robot.lcm_handler.joint_current_pos.copy()
   
    left_current_cart = robot.Kinematic_Model.left_arm_forward_kinematics(current_joint[0:7])    
    right_current_cart = robot.Kinematic_Model.right_arm_forward_kinematics(current_joint[7:14])
    
    # 左臂末端移动位置(0,0,0)->(0.1,0,0)->(0.1, 0.1,0)
    right_target_cart1 = right_current_cart.copy()
    right_target_cart1.translation = right_target_cart1.translation + [0, 0.2, 0]
    right_target_cart2 = right_target_cart1.copy()
    right_target_cart2.translation = right_target_cart2.translation + [0.1, 0, 0]
    robot.hybrid_force_movel_plan_right_target_cart_list = [right_target_cart1, right_target_cart2]
    
    robot.hybrid_force_movel_plan_left_target_cart_list = [left_current_cart, left_current_cart]
    
    # 左臂接触力设置 z_force = 10
    robot.hybrid_force_movel_plan_target_FT_data_list = [[0,0,0,0,0,0,  0,0,6.25,0,0,0],
                                                         ]

    time.sleep(1)
    robot.trajectory_segment_index = 0
    robot.Hybrid_Force_MoveL.whether_save_movel_position = 0
    robot.robot_hybrid_force_movel_to_target_cart(1)

    
