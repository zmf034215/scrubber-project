import time
from robot_model import robot_model
import numpy as np


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)

    # 设置运动控制接口到Kinematic_Model中
    robot.Force_Control.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)

    # 设置目标臂和起始位姿
    # arm = 'left'
    # start_pose = np.array([0.3, 0.45, 0.2, 1.8, 0.75, -1.8])  # [x, y, z, roll, pitch, yaw]

    arm = 'right'
    start_pose = np.array([0.3, -0.55, 0.2, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]

    robot.Force_Control.left_arm_target_FT_data = np.array([0, 0, 3, 0, 0, 0])
    robot.Force_Control.right_arm_target_FT_data = np.array([0, 0, 1, 0, 0, 0])
    hold_time= 0.5    #  运动到擦拭起始位置后，暂停保持当前位置的时间 秒
    wipe_direction=np.array([0.0, 1.0])  #  擦拭运动方向X、Y，左臂=[0.0, -1.0]，右臂=[0.0, 1.0]
    wipe_step=0.002   # 擦拭运动步长
    wipe_total_distance=0.35  # 擦拭运动距离
    
    robot.Force_Control.desktop_wiping_force_tracking_control(arm,start_pose, hold_time,wipe_direction, wipe_step, wipe_total_distance)

    # for cycle in range(3):  # 往返3次
    #     robot.Force_Control.desktop_wiping_force_tracking_control(arm, start_pose, hold_time, 
    #                                                             wipe_direction, wipe_step, wipe_total_distance)
    #     # 改变方向，回走
    #     wipe_direction = -wipe_direction
