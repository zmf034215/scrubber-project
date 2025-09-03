import time
from robot_model import robot_model
import numpy as np
import pinocchio as pin
import os

def test_by_position_control(robot:robot_model, arm, start_position, hold_time=0.5,  push_or_not=1, loop=1, rotation_or_not=0, velocity=0.2):
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
    :param velocity: 运动速度
    """

    # wipe_total_distance = np.array(wipe_total_distance)

    time.sleep(hold_time)
    # 运动到初始位姿
    print("开始移动到起始位姿...")
    success = robot.Kinematic_Model.move_to_start_pose(arm, start_position)
    
    if not success:
        print("❌ 起始位姿运动失败，程序终止。")
        exit()
    else:
        print("✅ 已到达起始位姿。")
    
    robot.MOVEJ.movej_plan_speed_max = velocity*3
        
    count = memory_loop()
    # print(f"已进行清洁循环次数： {count}")
    # if count >= loop :
    #     print(f"清洁循环已超过 {loop} 次，请清除循环次数，然后再次启动清洁程序！")
    #     exit()
    print("🧽 开始清洁...")
    current_joint_position = robot.lcm_handler.joint_current_pos.copy()
    target_push_joint1 = current_joint_position.copy()
    target_push_joint2 = current_joint_position.copy()
    if arm == "left":
        target_push_joint1[0] = current_joint_position[0] + np.pi/6
        target_push_joint1[3] = current_joint_position[3] + (-np.pi/6)
        target_push_joint2[0] = -np.pi/3
        # target_push_joint2[3] = current_joint_position[3] + np.pi/6
        target_push_joint2[3] = np.pi/2
    elif arm == "right" :
        target_push_joint1[7] = current_joint_position[7] + np.pi/6
        target_push_joint1[10] = current_joint_position[10] + (-np.pi/6)
        target_push_joint2[7] = -np.pi/3
        # target_push_joint2[10] = current_joint_position[10] + np.pi/6
        target_push_joint2[10] = np.pi/2
    else :
        target_push_joint1[0] = current_joint_position[0] + np.pi/6
        target_push_joint1[3] = current_joint_position[3] + (-np.pi/6)
        target_push_joint2[0] = -np.pi/3
        # target_push_joint2[3] = current_joint_position[3] + np.pi/6
        target_push_joint2[3] = np.pi/2
        target_push_joint1[7] = current_joint_position[7] + np.pi/6
        target_push_joint1[10] = current_joint_position[10] + (-np.pi/6)
        target_push_joint2[7] = -np.pi/3
        # target_push_joint2[10] = current_joint_position[10] + np.pi/6
        target_push_joint2[10] = np.pi/2

    current_joint_position = robot.lcm_handler.joint_current_pos.copy()
    robot.MOVEJ.moveJ2target(current_joint_position, target_push_joint2)
    time.sleep(0.1)

    while True:
        # robot.Kinematic_Model.move_relative(arm, wipe_total_distance)
        
        if rotation_or_not :
            # current_joint_position = robot.lcm_handler.joint_current_pos.copy()
            current_joint_position = target_push_joint2
            if arm == "left":
                target_joint_position1 = current_joint_position.copy()
                target_joint_position1[2] = current_joint_position[2] + (-np.pi/12)
                target_joint_position2 = current_joint_position.copy()
                target_joint_position2[2] = current_joint_position[2] + np.pi/12
            elif arm =="right" :
                target_joint_position1 = current_joint_position.copy()
                target_joint_position1[9] = current_joint_position[9] + np.pi/12
                target_joint_position2 = current_joint_position.copy()
                target_joint_position2[9] = current_joint_position[9] + (-np.pi/12)
            else :
                target_joint_position1 = current_joint_position.copy()
                target_joint_position1[2] = current_joint_position[2] + (-np.pi/12)
                target_joint_position2 = current_joint_position.copy()
                target_joint_position2[2] = current_joint_position[2] + np.pi/12
                target_joint_position1[9] = current_joint_position[9] + np.pi/12
                target_joint_position2[9] = current_joint_position[9] + (-np.pi/12)
            robot.MOVEJ.moveJ2target(current_joint_position, target_joint_position1)
            time.sleep(0.1)
            robot.MOVEJ.moveJ2target(target_joint_position1,current_joint_position)
            time.sleep(0.1)
            robot.MOVEJ.moveJ2target(current_joint_position, target_joint_position2)
            time.sleep(0.1)
            robot.MOVEJ.moveJ2target(target_joint_position2,current_joint_position)
            time.sleep(0.1)
        # robot.Kinematic_Model.move_relative(arm, -(wipe_total_distance))
        if push_or_not :
            current_joint_position = robot.lcm_handler.joint_current_pos.copy()
            # robot.MOVEJ.moveJ2target(current_joint_position, target_push_joint1)
            robot.MOVEJ.moveJ2target(target_push_joint2, target_push_joint1)
            time.sleep(0.1)
            robot.MOVEJ.moveJ2target(target_push_joint1,target_push_joint2)
            time.sleep(0.1)

        # 循环记录
        count += 1
        with open("count.txt", "w") as f:
            f.write(str(count))
        print(f"已进行清洁循环次数：  {count}")
        if count >= loop :
            print(f"已清洁循环 {loop} 次，结束清洁！！")
            exit()
        
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
    return count
    


def reset_count(path="count.txt"):
    # 重置计数器
    with open(path, "w") as f:
        f.write("0")


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)
    # reset_count()

    # 设置运动控制接口到Kinematic_Model中
    robot.Force_Control.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)

    # 设置目标臂和起始位姿
    # arm = 'left'
    # arm = 'right'
    arm = 'both'
    start_position = [-0.1,  0.2,  0, 0.3,  0, 0, 0,
                      -0.1, -0.2, -0, 0.3, -0, 0, 0]  # 关节角度 弧度
    
    hold_time= 0.5    #  运动到擦拭起始位置后，暂停保持当前位置的时间 秒
    loop = 77
    push_or_not = 1   #  0： 不进行推拉， 1： 进行推拉
    rotation_or_not = 1    # 0: 不进行扭转，1： 进行扭转
    velocity = 0.6   # 关节转动速度

    count = memory_loop()
    print(f"已进行清洁循环次数： {count}")
    if count >= loop :
        print(f"清洁循环已超过 {loop} 次，请清除循环次数，然后再次启动清洁程序！")
        exit()
  
    test_by_position_control(robot, arm, start_position, hold_time, push_or_not, loop, rotation_or_not, velocity=velocity)
    
    # 采用纯位置控制实现洗地机测试程度

   
