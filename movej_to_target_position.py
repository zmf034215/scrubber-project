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
    robot.movej_plan_target_position_list = [
                                                [-0.6412878036499023, 0.6774682998657227, 0.2993779182434082, -1.502809762954712, 0.04234027862548828, -0.027610667049884796, -0.0347219817340374,
                                                 -0.6412878036499023, -0.6774682998657227, -0.2993779182434082, 1.502809762954712, -0.04234027862548828, 0.027610667049884796, 0.0347219817340374] + hand_home_pos + [0, 0, 0, 0],
                                                [-0.3, 0.7, 1.5, -1.27, -2.2, 0.2, 0,
                                                 -0.3, -0.7, -1.5, 1.27, 2.2, -0.2, 0] + hand_home_pos + [0, 0, 0, 0]
                                            ]
    time.sleep(1)

    robot.robot_movej_to_target_position()
