import time
from robot_model import robot_model
import numpy as np


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)

    robot.Force_Control.left_arm_target_FT_data = np.array([0, 80, 0, 0, 0, 0])
    robot.Force_Control.right_arm_target_FT_data = np.array([0, -80, 0, 0, 0, 0])
    robot.Force_Control.constant_force_tracking_control()
