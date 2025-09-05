import time
import numpy as np
import math
from robot_model import robot_model


if __name__ == "__main__":
    robot = robot_model()
    # 该参数设置为0 是不进行碰撞检测
    robot.Collision_Detection.collision_detection_level = 0
    # 该参数设置为0，表示不启用阻抗控制
    robot.lcm_handler.impedance_control.impedance_control_flag = 1
    hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
    hand_home_pos = list(hand_home_pos / 180 * np.pi)
    robot.movej_plan_target_position_list = [
                                                [0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0] + hand_home_pos + [0, 0, 0, 0],
                                                [-0.1,  0.2,  0.5, 0.3,  0.5, 0, 0,
                                                 -0.1, -0.2, -0.5, 0.3, -0.5, 0, 0] + hand_home_pos + [0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0] + hand_home_pos + [0, 0, 0, 0],
                                            ]
    time.sleep(1)

    robot.robot_movej_to_target_position()
