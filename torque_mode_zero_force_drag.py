import time
import numpy as np
import math
from robot_model import robot_model
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 请输入保存的CSV文件名，例如：python torque_mode_zero_force_drag.py mylog.csv")
        sys.exit(1)

    csv_filename = sys.argv[1]

    robot = robot_model()
    robot.Zero_Force_Drag.torque_mode_zero_force_drag(csv_filename)
    
