import time
from robot_model import robot_model
import numpy as np
import pinocchio

if __name__ == "__main__":
    # 实例化机器人模型
    robot = robot_model()
    time.sleep(1)  # 等待初始化完成

    # 设置运动控制接口
    robot.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)

    # 设置目标臂和起始位姿
    # arm = 'left'
    # start_pose = np.array([0.3, 0.45, 0.2, 1.8, 0.75, -1.8])  # [x, y, z, roll, pitch, yaw]

    arm = 'right'
    start_pose = np.array([0.3, -0.45, 0.2, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]

    from scipy.spatial.transform import Rotation as R

    q_current = robot.lcm_handler.joint_current_pos[7:14]
    pose_r = robot.Kinematic_Model.right_arm_forward_kinematics(q_current)

    R_r = pose_r.rotation
    t_r = pose_r.translation

    # 测试逆解
    robot.Kinematic_Model.right_arm_inverse_kinematics(R_r, t_r, q_current)

    if robot.Kinematic_Model.right_arm_inverse_kinematics_solution_success_flag:
        print("✅ 正逆解一致，逆解成功！")
    else:
        print("❌ 正逆解失败，请检查模型或参数")


    


