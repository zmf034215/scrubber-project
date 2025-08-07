import time
from robot_model import robot_model
import numpy as np
import pinocchio as pin

def desktop_wiping_interface(robot:robot_model, arm, start_pose, hold_time, target_FT, 
                             loop = 1, wipe_direction = np.array([0,0]), wipe_total_distance = 0.1,
                             rotation_direction = 0, rotation_deg = 0):
    """
    实现桌面擦拭的接口函数
    :param robot: 机器人类实例
    :param arm: 选择擦拭的臂
    :param start_pose: 起始位姿
    :param hold_time: 保持位姿时间
    :param target_FT: 目标力
    :param loop: 循环次数
    :param wipe_direction: 擦拭方向
    :param wipe_total_distance: 擦拭距离
    :rotation_direction: 旋转方向
    :rotation_deg: 旋转角度
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
    target_FT_data = None


    if arm == 'right':
        # 选择右臂运动
        right_target_cart.translation = start_pose[:3]
        right_target_cart.rotation = pin.rpy.rpyToMatrix(start_pose[3:] + np.array([0, 0, rotation_direction * rotation_deg]))
        right_target_cart.translation[:2] += wipe_direction * wipe_total_distance

        left_target_cart = left_current_cart.copy()
        target_FT_data = [0,0,0,0,0,0] + target_FT
    else:
        left_target_cart.translation = start_pose[:3]
        left_target_cart.rotation = pin.rpy.rpyToMatrix(start_pose[3:] + np.array([0, 0, rotation_direction * rotation_deg]))
        left_target_cart.translation += wipe_direction * wipe_total_distance

        right_target_cart = right_current_cart.copy()

        target_FT_data = target_FT + [0,0,0,0,0,0] 

    # hybrid_movel 参数赋值
    robot.hybrid_force_movel_plan_left_target_cart_list = [left_target_cart, left_current_cart]
    robot.hybrid_force_movel_plan_right_target_cart_list = [right_target_cart, right_current_cart]
    robot.hybrid_force_movel_plan_target_FT_data_list = [target_FT_data]
    # 擦桌子主任务
    for i in range(loop):
        print("开始擦桌子第{}次...".format(i+1))
        robot.robot_hybrid_force_movel_to_target_cart(0)
        time.sleep(0.5)
        robot.trajectory_segment_index = 0

    robot.Kinematic_Model.back_to_start_pose(arm, start_pose)
    print("擦桌子任务完成。")





if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)

    # 设置运动控制接口到Kinematic_Model中
    robot.Force_Control.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)

    # 设置左右臂
    arm = 'right'
    start_pose = np.array([0.3, -0.55, 0.2, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]

    # 设置目标力
    left_arm_target_FT_data = [0, 0, 3, 0, 0, 0]
    right_arm_target_FT_data = [0, 0, 6.25, 0, 0, 0]
    hold_time= 0.5    #  运动到擦拭起始位置后，暂停保持当前位置的时间 秒
    wipe_direction=np.array([0.0, 1.0])  #  擦拭运动方向X、Y，左臂=[0.0, -1.0]，右臂=[0.0, 1.0]
    # wipe_step=0.002   # 擦拭运动步长
    wipe_total_distance=0.35  # 擦拭运动距离

    # 旋转方向 
    rotation_direction = -1  # 1代表正方向，-1代表反方向，0表示不转动
    rotation_deg = 1.57    # 旋转角度(弧度制)

    desktop_wiping_interface(robot, arm, start_pose, hold_time, right_arm_target_FT_data, 
                             loop=3, wipe_direction=wipe_direction, wipe_total_distance=wipe_total_distance,
                             rotation_direction=rotation_direction, rotation_deg=rotation_deg)

    
    
