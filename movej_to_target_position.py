import time
import numpy as np
import math
from robot_model import robot_model



if __name__ == "__main__":
    robot = robot_model()
    hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
    hand_home_pos = list(hand_home_pos / 180 * np.pi)
    robot.movej_plan_target_position_list = [
                                                
                                                [
                                                    0,	0,	0,	0,	0,	0,  0,
                                                    0,	0,	0,	0,	0,	0,  0,
                                                    0,	0,	0,	0,	0,	0,
                                                    0,	0,	0,	0,	0,	0,
                                                    0,	0,	0,	0
                                                ],
                                                [
                                                    -1.47157647066741,	1.00427773317030,	0.615773156414501,	-1.02514569327143,	0.237506725899294,	0,	0,
                                                    -1.47157647066741,	-1.00427773317030,	-0.615773156414501,	1.02514569327143,	-0.237506725899294,	0,	0,
                                                    0,	0,	0,	0,	0,	0,
                                                    0,	0,	0,	0,	0,	0,
                                                    0,	0,	0,	0
                                                ]
                                            ]
    time.sleep(1)
    robot.robot_movej_to_target_position()
