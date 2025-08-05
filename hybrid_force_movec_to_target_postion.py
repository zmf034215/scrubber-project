import time
import numpy as np
import math
from robot_model import robot_model



if __name__ == "__main__":
    robot = robot_model()
    # 该参数设置为0 是不进行碰撞检测
    robot.Collision_Detection.collision_detection_level = 0
    hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
    hand_home_pos = list(hand_home_pos / 180 * np.pi)
    # robot.hybrid_force_movec_plan_target_joint_list = [
    #                                             [-0.6412878036499023, 0.6774682998657227, 0.2993779182434082, -1.502809762954712, 0.04234027862548828, -0.027610667049884796, -0.0347219817340374, 
    #                                              -0.6412878036499023, -0.6774682998657227, -0.2993779182434082, 1.502809762954712, -0.04234027862548828, 0.027610667049884796, 0.0347219817340374] + hand_home_pos + [-math.pi/8, 0, 0, -math.pi/16],

    #                                             [-1.0877361297607422, 0.681267261505127, 0.4956197738647461, -0.8803613185882568, 0.0423884391784668, -0.02760806493461132, -0.03472469002008438, 
    #                                              -1.0877361297607422, -0.681267261505127, -0.4956197738647461, 0.8803613185882568, -0.0423884391784668, 0.02760806493461132, 0.03472469002008438] + hand_home_pos + [-math.pi/8, 0, 0, -math.pi/16],
    #                                         ]
    # robot.hybrid_force_movec_plan_target_FT_data_list = [0,0,0,0,0,0,0,0,10,0,0,0]

    # time.sleep(1)
    # robot.Hybrid_Force_MoveC.whether_save_movec_position = 0
    # robot.robot_hybrid_force_movec_to_target_joint()


    # 只对右臂在xy平面进行运动
    current_joint = robot.lcm_handler.joint_current_pos.copy()
    left_current_cart = robot.Kinematic_Model.forward_kinematics_left_arm(current_joint[0:7])    
    right_current_cart = robot.Kinematic_Model.forward_kinematics_right_arm(current_joint[7:14])
    robot.hybrid_force_movec_plan_left_target_cart_list = [left_current_cart,left_current_cart]
    
    # 右臂末端移动位置(0,0,0)->(0.1,0.1,0)->(0.2,0,0)
    right_middle_cart = right_current_cart.copy()
    right_middle_cart.translation = right_middle_cart.translation + [0.1, 0.1, 0]
    right_target_cart = right_current_cart.copy()
    right_target_cart.translation = right_target_cart.translation + [0.2, 0, 0]
    robot.hybrid_force_movec_plan_right_target_cart_list = [right_middle_cart,right_target_cart]
    
    # 右臂接触力设置 z_force = 10
    robot.hybrid_force_movec_plan_target_FT_data_list = [0,0,0,0,0,0,  0,0,10,0,0,0]

    time.sleep(1)

    robot.Hybrid_Force_MoveC.whether_save_movec_position = 0
    robot.robot_hybrid_force_movec_to_target_cart(0.5)

    
