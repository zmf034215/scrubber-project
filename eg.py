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

    print("🤖 开始移动到起始位姿...")
    success = robot.Kinematic_Model.move_to_start_pose(arm, start_pose)
    if not success:
        print("❌ 起始位姿运动失败，程序终止。")
        exit()
    else:
        print("✅ 已到达起始位姿。")
    time.sleep(0.5)    

                                # ****************left  arm
    # current_posi1 = robot.lcm_handler.joint_current_pos.copy()
    # start_pose1 = np.array([0.3, 0.45, 0.0, 1.8, 0.75, -1.8])  # [x, y, z, roll, pitch, yaw]
    # R = pinocchio.rpy.rpyToMatrix(start_pose1[3:])
    # robot.Kinematic_Model.left_arm_inverse_kinematics(R, start_pose1[:3], np.array(current_posi1[:7]))
    # target_posi1 = np.array(robot.Kinematic_Model.left_arm_interpolation_result.tolist() + current_posi1[7:].tolist())
    # robot.MOVEL.moveL2targetjointposition(current_posi1,target_posi1)
    # time.sleep(2)

    # start_pose2 = np.array([0.3, 0.15, 0.0, 1.8, 0.75, -1.8])  # [x, y, z, roll, pitch, yaw]
    # R = pinocchio.rpy.rpyToMatrix(start_pose2[3:])
    # robot.Kinematic_Model.left_arm_inverse_kinematics(R, start_pose2[:3], np.array(target_posi1[:7]))
    # if not robot.Kinematic_Model.left_arm_inverse_kinematics_solution_success_flag :
    #     print("ink false")
    #     # exit()
    # target_posi2 = np.array(robot.Kinematic_Model.left_arm_interpolation_result.tolist() + target_posi1[7:].tolist())
    # robot.MOVEL.moveL2targetjointposition(target_posi1,target_posi2)
    # time.sleep(2)
    # robot.Kinematic_Model.move_relative(arm, np.array([0, 0, 0.02]))
    # robot.Kinematic_Model.back_to_start_pose(arm,start_pose)
    

                                # *************right  arm
    current_posi1 = robot.lcm_handler.joint_current_pos.copy()
    start_pose1 = np.array([0.3, -0.45, 0.15, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]
    R = pinocchio.rpy.rpyToMatrix(start_pose1[3:])
    robot.Kinematic_Model.right_arm_inverse_kinematics(R, start_pose1[:3], np.array(current_posi1[7:14]))
    target_posi1 = np.array(current_posi1[:7].tolist() + robot.Kinematic_Model.right_arm_interpolation_result.tolist() + current_posi1[14:].tolist())
    robot.MOVEL.moveL2targetjointposition(current_posi1,target_posi1)
    time.sleep(2)

    # start_pose2 = np.array([0.3, -0.25, 0.1, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]
    # R = pinocchio.rpy.rpyToMatrix(start_pose2[3:])
    # robot.Kinematic_Model.right_arm_inverse_kinematics(R, start_pose2[:3], np.array(target_posi1[7:14]))
    # target_posi2 = np.array(target_posi1[:7].tolist() + robot.Kinematic_Model.right_arm_interpolation_result.tolist() + target_posi1[14:].tolist())
    # robot.MOVEL.moveL2targetjointposition(target_posi1,target_posi2)
    # time.sleep(2)


                                #########
    wipe_direction=np.array([0.0, 1.0])  #  擦拭运动方向X、Y，左臂=[0.0, -1.0]，右臂=[0.0, 1.0]
    wipe_step=0.002   # 擦拭运动步长
    wipe_total_distance=0.25  # 擦拭运动距离
    print("🧽 开始擦拭...")
    # wipe_steps = int(wipe_total_distance / wipe_step)
    # dx = wipe_direction[0] * wipe_step
    # dy = wipe_direction[1] * wipe_step
    
    # for i in range(wipe_steps):
                          
    #     dz = 0.000

    #     delta = np.array([dx, dy, dz])
    #     success = robot.Kinematic_Model.move_relative(arm, delta)
    #     if not success :
    #         print("❌ 超出机械臂可达空间！！！")
    #         exit()
    #     time.sleep(2.0 / 1000.0)

    dx = wipe_direction[0] * wipe_total_distance
    dy = wipe_direction[1] * wipe_total_distance
    dz = 0.0
    delta = np.array([dx, dy, dz])
    success = robot.Kinematic_Model.move_relative(arm, delta)
    print("ca shi****")
    if not success :
        print("❌ 超出机械臂可达空间！！！")
        exit()
            
    print(">>> 全部完成，抬升 2 cm")    
    robot.Kinematic_Model.move_relative(arm, np.array([0, 0, 0.02]))
    time.sleep(2)
    robot.Kinematic_Model.back_to_start_pose(arm,start_pose) 
    print("✅ 擦拭任务完成。")

    


