import time
import numpy as np
import math
from robot_model import robot_model



if __name__ == "__main__":
    robot = robot_model()
    time.sleep(0.1)
    print(robot.lcm_handler.joint_current_pos)
