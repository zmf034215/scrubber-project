import time
import numpy as np
import math
from robot_model import robot_model
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 请提供CSV文件路径，例如：python get_csv_position_and_interpolation.py your_csv_file.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    robot = robot_model()
    time.sleep(1)

    speed_scale = 1  # 设置csv轨迹复现的速度调节参数
    max_recycle = 3
    count = 0
    while count < max_recycle:
        robot.get_csv_position_and_interpolation(csv_path, speed_scale)
        count+=1
        print(f"count = {count}")
        time.sleep(0.5)

