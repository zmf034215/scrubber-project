import time
from robot_model import robot_model
import numpy as np
import pinocchio as pin
import os

def test_by_position_control(robot:robot_model, arm, start_pose, hold_time, wipe_direction, wipe_step, wipe_total_distance, loop, rotation_direction, rotation_deg):
    # 采用位置控制实现洗地机测试程度
    """
    :param robot: 机器人模型
    :param arm: 选择擦拭臂
    :param start_pose: 起始位姿
    :param hold_time: 运动到擦拭起始位置后，暂停保持当前位置的时间 秒    
    :param wipe_direction: 擦拭运动方向X、Y，左臂=[0.0, -1.0]，右臂=[0.0, 1.0]
    :param wipe_step: 擦拭运动步长
    :param wipe_total_distance: 擦拭运动距离
    :param loop: 擦拭循环次数
    :param rotation_direction: 旋转方向，1为顺时针，-1为逆时针
    :param rotation_deg: 旋转角度
    """

    time.sleep(hold_time)
    # 运动到初始位姿
    print("开始移动到起始位姿...")
    success = robot.Kinematic_Model.move_to_start_pose(arm, start_pose)
    
    
    if not success:
        print("❌ 起始位姿运动失败，程序终止。")
        exit()
    else:
        print("✅ 已到达起始位姿。")

    robot.Force_Control.arm_FT = arm
    right_target_cart = pin.SE3.Identity()
    left_target_cart = pin.SE3.Identity()
    right_current_cart = robot.Kinematic_Model.right_arm_forward_kinematics(robot.lcm_handler.joint_current_pos[7:14])
    left_current_cart = robot.Kinematic_Model.left_arm_forward_kinematics(robot.lcm_handler.joint_current_pos[:7])
    # target_FT_data = None     

    current_joint_position = robot.lcm_handler.joint_current_pos.copy()

    if arm == 'right':
        right_target_cart.translation = start_pose[:3]
        right_target_cart.rotation = pin.rpy.rpyToMatrix(start_pose[3:]+[0,0,rotation_direction * rotation_deg])
        # 直接在当前目标的rpy下加上z轴的旋转角度
        right_target_cart.translation[:2] += wipe_direction * wipe_total_distance
        left_target_cart = left_current_cart.copy()
        

    else:
        left_target_cart.translation = start_pose[:3]
        left_target_cart.rotation = pin.rpy.rpyToMatrix(start_pose[3:]+[0,0,rotation_direction * rotation_deg])
        # 直接在当前目标的rpy下加上z轴的旋转角度
        left_target_cart.translation[:2] += wipe_direction * wipe_total_distance
        right_target_cart = right_current_cart.copy()

    # robot.movel_plan_target_position_list = [target_joint_position, current_joint_position]

    # 开始擦拭
    print("🧽 开始擦拭...")
    for i in range(loop):
        print("开始擦桌子第{}次...".format(i+1))
        # robot.robot_movel_to_target_position()
        robot.MOVEL.moveL2target(left_current_cart, left_target_cart, right_current_cart, right_target_cart, current_joint_position)
        # 如果逆解失败
        if (robot.Kinematic_Model.left_arm_inverse_kinematics_solution_success_flag and robot.Kinematic_Model.right_arm_inverse_kinematics_solution_success_flag) == False:
            return
        
        current_joint_position = robot.lcm_handler.joint_current_pos.copy()
        # 回走
        robot.MOVEL.moveL2target(left_target_cart, left_current_cart, right_target_cart, right_current_cart, current_joint_position)
        time.sleep(0.5)
        # robot.trajectory_segment_index = 0
        memory_loop()

def memory_loop(path="count.txt"):
    if not os.path.exists(path):
        # 不存在文件创建文件
        with open(path, "w") as f:
            f.write("0")

    with open(path, "r+") as f:
        count = f.read().strip()
        if count.isdigit():
            # 如果确定里面存放的是数字，则读取数字
            count = int(count)
        else:
            f.write('0')
            count = 0

    count += 1
    with open(path, "w") as f:
        f.write(str(count))


def reset_count(path="count.txt"):
    # 重置计数器
    with open(path, "w") as f:
        f.write("0")


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)
    reset_count()

    # 设置运动控制接口到Kinematic_Model中
    robot.Force_Control.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)

    # 设置目标臂和起始位姿
    # arm = 'left'
    # start_pose = np.array([0.3, 0.45, 0.2, 1.8, 0.75, -1.8])  # [x, y, z, roll, pitch, yaw]

    arm = 'right'
    start_pose = np.array([0.3, -0.55, 0.2, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]

    # robot.Force_Control.left_arm_target_FT_data = np.array([0, 0, 3, 0, 0, 0])
    # robot.Force_Control.right_arm_target_FT_data = np.array([0, 0, 2, 0, 0, 0])
    hold_time= 0.5    #  运动到擦拭起始位置后，暂停保持当前位置的时间 秒
    wipe_direction=np.array([-1 ,  0])  #  擦拭运动方向X、Y，左臂=[0.0, -1.0]，右臂=[0.0, 1.0]
    wipe_step=0.002   # 擦拭运动步长
    wipe_total_distance=0.3  # 擦拭运动距离
    loop = 7
    rotation_direction = 1
    rotation_deg = 1
    
    test_by_position_control(robot, arm, start_pose, hold_time, wipe_direction, wipe_step, wipe_total_distance, loop, rotation_direction, rotation_deg)
    # robot.Force_Control.desktop_wiping_force_tracking_control(arm,start_pose, hold_time,wipe_direction, wipe_step, wipe_total_distance,
    #                                                           loop, rotation_direction, rotation_deg)

    # 采用纯位置控制实现洗地机测试程度


    # for cycle in range(3):  # 往返3次
    #     robot.Force_Control.desktop_wiping_force_tracking_control(arm, start_pose, hold_time, 
    #                                                             wipe_direction, wipe_step, wipe_total_distance)
    #     # 改变方向，回走
    #     wipe_direction = -wipe_direction
