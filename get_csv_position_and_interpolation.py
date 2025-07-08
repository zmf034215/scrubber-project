import time
import numpy as np
import math
from robot_model import robot_model

if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)

    robot.get_csv_position_and_interpolation()
