import time
from robot_model import robot_model
import numpy as np


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)

    ## 采用纯逆解方案 具有锁轴拖动的功能 
    robot.Force_Control.force_sensor_drag_teach_whether_use_IK = True
    robot.Force_Control.left_arm_force_sensor_drag_teach_lock_axis_sign = np.array([1, 0, 0, 0, 0, 0])
    robot.Force_Control.right_arm_force_sensor_drag_teach_lock_axis_sign = np.array([1, 0, 0, 0, 0, 0])

    
    robot.Force_Control.force_sensor_drag_teach()
