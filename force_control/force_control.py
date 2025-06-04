from lcm_handler import LCMHandler
from force_control.force_control_data_cal import Force_Control_Data_Cal
import numpy as np
import time
from copy import deepcopy
from robot_kinematics_and_dynamics_models.Kinematic_Model import Kinematic_Model
from trajectory_plan.moveJ import MOVEJ
import threading
import math
import csv
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


class Force_Control():
    def __init__(self, LCMHandler, force_control_data_cal):
        # lcm
        self.lcm_handler = LCMHandler
        self.force_control_data = force_control_data_cal
        self.Kinematic_Model = Kinematic_Model()

        # 机器人笛卡尔空间下的位置、速度、加速度  用于实现导纳控制的拖动
        self.force_control_left_arm_current_cart = pin.SE3.Identity()
        self.force_control_right_arm_current_cart = pin.SE3.Identity()

        self.left_arm_target_cart_position = np.array([0, 0, 0, 0, 0, 0])
        self.left_arm_effector_current_speed = np.array([0, 0, 0, 0, 0, 0])
        self.left_arm_effector_current_acc = np.array([0, 0, 0, 0, 0, 0])
        self.left_arm_effector_pre_speed = np.array([0, 0, 0, 0, 0, 0])
        self.left_arm_effector_pre_position = np.array([0, 0, 0, 0, 0, 0])
        self.left_arm_effector_pre_acc = np.array([0, 0, 0, 0, 0, 0])

        self.right_arm_target_cart_position = np.array([0, 0, 0, 0, 0, 0])
        self.right_arm_effector_current_speed = np.array([0, 0, 0, 0, 0, 0])
        self.right_arm_effector_current_acc = np.array([0, 0, 0, 0, 0, 0])
        self.right_arm_effector_pre_speed = np.array([0, 0, 0, 0, 0, 0])
        self.right_arm_effector_pre_position = np.array([0, 0, 0, 0, 0, 0])
        self.right_arm_effector_pre_acc = np.array([0, 0, 0, 0, 0, 0])

        # 是否需要使用逆解方案来实现拖动以及恒力跟踪标志位 0不使用 1使用
        self.force_sensor_drag_teach_whether_use_IK = True
        self.constant_force_tracking_control_whether_use_IK = True

        # 是否存储拖动/恒力跟踪产生的期望位置曲线标志位 0 不保存 1 保存
        self.whether_save_drag_position = False
        self.whether_save_constant_force_track_control_position = False

        # 恒力跟踪控制中的期望力设置   
        self.left_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])
        self.right_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])

        # 导纳控制实现恒力跟踪的系数设置   这个地方由于代码写的有问题 需要重新调节这个参数 可以先调节单臂的参数 然后 按照单臂的参数写 
        self.left_arm_constant_force_M = np.array([5, 5, 5, 5, 5, 5])
        self.left_arm_constant_force_B = np.array([500, 500, 500, 500, 500, 500])
        self.right_arm_constant_force_M = np.array([5, 5, 5, 5, 5, 5])
        self.right_arm_constant_force_B = np.array([500, 500, 500, 500, 500, 500])

        # 导纳控制实现拖动示教的系数设置   这个地方由于代码写的有问题 需要重新调节这个参数 可以先调节单臂的参数 然后 按照单臂的参数写 
        self.left_arm_admittance_control_M = np.array([0.1, 0.1, 0.1, 100, 100, 100])
        self.left_arm_admittance_control_B = np.array([0.05, 0.05, 0.05, 100, 100, 100])

        self.right_arm_admittance_control_M = np.array([0.1, 0.1, 0.1, 100, 100, 100])
        self.right_arm_admittance_control_B = np.array([0.05, 0.05, 0.05, 100, 100, 100])

        self.left_arm_admittance_control_M_end_cartesian_space_plan = np.array([0.001, 0.001, 0.001, 10, 10, 10])
        self.left_arm_admittance_control_B_end_cartesian_space_plan = np.array([0.15, 0.15, 0.15, 5, 5, 5])


        self.right_arm_admittance_control_M_end_cartesian_space_plan = np.array([0.001, 0.001, 0.001, 10, 10, 10])
        self.right_arm_admittance_control_B_end_cartesian_space_plan = np.array([0.15, 0.15, 0.15, 5, 5, 5])

        self.interpolation_period = 2
        self.joint_target_position = None

        # 锁轴拖动 锁轴恒力跟踪 

    def force_sensor_drag_teach(self):
        self.joint_target_position = self.lcm_handler.joint_current_pos
        self.last_joint_target_position = self.joint_target_position

        while(1):
            start_time = time.time()  # 记录循环开始的时间
            if(self.force_sensor_drag_teach_whether_use_IK):
                ## 拖动示教使用逆解方案
                # 左臂的拖动处理
                Ftmp = math.sqrt(self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[0] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[1] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[2] ** 2) 
                Mtmp = math.sqrt(self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[3] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[4] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[5] ** 2)
                # print("请注意进入拖动状态 Ftmp = {}".format(Ftmp))
                # print("请注意进入拖动状态 Mtmp = {}".format(Mtmp))

                # 这个参数 需要进一步调节 直到拖动时感觉不出明显的卡顿为止
                if (Ftmp > 1) or (Mtmp > 0.5):
                    self.force_control_left_arm_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movel_plan_current_joint_position[:7]))
                    self.left_arm_target_cart_position = deepcopy(self.force_control_left_arm_current_cart.translation)
                    self.left_arm_target_cart_pose = deepcopy(self.force_control_left_arm_current_cart.rotation)                    
                    self.left_arm_effector_pre_position = self.left_arm_target_cart_position

                    # 导纳控制输出笛卡尔空间下的速度
                    self.left_arm_effector_current_acc = (self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system - self.left_arm_admittance_control_B_end_cartesian_space_plan @ self.left_arm_effector_pre_speed) / self.left_arm_admittance_control_M_end_cartesian_space_plan 
                    self.left_arm_effector_current_acc = 0.5 * self.left_arm_effector_current_acc + 0.5 * self.left_arm_effector_pre_acc

                    self.left_arm_effector_current_speed = (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc) * (self.interpolation_period / 1000)
                    self.left_arm_effector_current_speed = 0.5 * self.left_arm_effector_current_speed + 0.5 * self.left_arm_effector_pre_speed

                    # 将计算的位置和姿态对应的速度值 积分成为笛卡尔空间下的位置
                    self.left_arm_target_cart_position = self.left_arm_target_cart_position + self.left_arm_effector_current_speed[:3] * (self.interpolation_period / 1000)
                    self.left_arm_target_cart_position = 0.015 * self.left_arm_target_cart_position + 0.985 * self.left_arm_effector_pre_position
                    self.left_arm_effector_pre_position = self.left_arm_target_cart_position

                    # 计算纯笛卡尔空间下的姿态
                    omega = self.left_arm_effector_current_speed[3:6] * self.interpolation_period / 1000
                    omega_norm = np.linalg.norm(omega)

                    if omega_norm > 1e-5:  
                        axis = omega / omega_norm  
                        sx = math.sin(omega_norm)
                        cx = math.cos(omega_norm)
                        v = 1 - cx
                        dR = np.array([[axis[0] * axis[0] * v + cx, axis[0] * axis[1] * v - axis[2] * sx, axis[0] * axis[2] * v + axis[1] * sx], 
                                    [axis[0] * axis[1] * v + axis[2] * sx, axis[1] * axis[1] * v + cx, axis[1] * axis[2] * v - axis[0] * sx], 
                                    [axis[0] * axis[2] * v - axis[1] * sx, axis[1] * axis[2] * v + axis[0] * sx, axis[2] * axis[2] * v + cx]])

                    else:
                        dR = np.eye(3)                   

                    self.left_arm_target_cart_pose = dR @ self.left_arm_target_cart_pose
                    self.left_arm_target_cart_pose_quat = R.from_matrix(self.left_arm_target_cart_pose).as_quat()

                    # 左臂逆解 逆解 逆解 
                    self.Kinematic_Model.left_arm_inverse_kinematics(self.left_arm_target_cart_pose, self.left_arm_target_cart_position, self.joint_target_position[:7])

                    if (self.Kinematic_Model.left_arm_inverse_kinematics_solution_success_flag):
                        ## 导纳控制需要的参数赋值
                        self.left_arm_effector_pre_acc = self.left_arm_effector_current_acc
                        self.left_arm_effector_pre_speed = self.left_arm_effector_current_speed

                        self.joint_target_position[:7] = self.Kinematic_Model.left_arm_interpolation_result
                    else:
                        self.joint_target_position[:7] = self.last_joint_target_position[:7]
                else:
                    self.joint_target_position[:7] = self.last_joint_target_position[:7]


                
                # 右臂的拖动处理
                Ftmp_right = math.sqrt(self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[0] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[1] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[2] ** 2) 
                Mtmp_right = math.sqrt(self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[3] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[4] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[5] ** 2)
                # print("请注意进入拖动状态 Ftmp_right = {}".format(Ftmp_right))
                # print("请注意进入拖动状态 Ftmp_right = {}".format(Ftmp_right))

                if (Ftmp_right > 1) or (Mtmp_right > 0.5):
                    self.force_control_right_arm_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movel_plan_current_joint_position[7:14]))
                    self.right_arm_target_cart_position = deepcopy(self.force_control_right_arm_current_cart.translation)
                    self.right_arm_target_cart_pose = deepcopy(self.force_control_right_arm_current_cart.rotation)                    
                    self.right_arm_effector_pre_position = self.right_arm_target_cart_position


                    # 导纳控制输出笛卡尔空间下的速度
                    self.right_arm_effector_current_acc = (self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system - self.right_arm_admittance_control_B_end_cartesian_space_plan @ self.right_arm_effector_pre_speed) / self.right_arm_admittance_control_M_end_cartesian_space_plan 
                    self.right_arm_effector_current_acc = (0.5 * self.right_arm_effector_current_acc + 0.5 * self.right_arm_effector_pre_acc)

                    self.right_arm_effector_current_speed = (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc) * (self.interpolation_period / 1000)
                    self.right_arm_effector_current_speed = 0.5 * self.right_arm_effector_current_speed + 0.5 * self.right_arm_effector_pre_speed

                    # 将计算的位置和姿态对应的速度值 积分成为笛卡尔空间下的位置
                    self.right_arm_target_cart_position = self.right_arm_target_cart_position + self.right_arm_effector_current_speed[:3] * (self.interpolation_period / 1000)
                    self.right_arm_target_cart_position = 0.015 * self.right_arm_target_cart_position + 0.985 * self.right_arm_effector_pre_position
                    self.right_arm_effector_pre_position = self.right_arm_target_cart_position

                    # 计算纯笛卡尔空间下的姿态
                    omega = self.right_arm_effector_current_speed[3:6] * self.interpolation_period / 1000
                    omega_norm = np.linalg.norm(omega)

                    if omega_norm > 1e-5:  
                        axis = omega / omega_norm  
                        sx = math.sin(omega_norm)
                        cx = math.cos(omega_norm)
                        v = 1 - cx
                        dR = np.array([[axis[0] * axis[0] * v + cx, axis[0] * axis[1] * v - axis[2] * sx, axis[0] * axis[2] * v + axis[1] * sx], 
                                    [axis[0] * axis[1] * v + axis[2] * sx, axis[1] * axis[1] * v + cx, axis[1] * axis[2] * v - axis[0] * sx], 
                                    [axis[0] * axis[2] * v - axis[1] * sx, axis[1] * axis[2] * v + axis[0] * sx, axis[2] * axis[2] * v + cx]])

                    else:
                        dR = np.eye(3)

                    self.right_arm_target_cart_pose = dR @ self.right_arm_target_cart_pose
                   
                    # 左臂逆解 逆解 逆解 
                    self.Kinematic_Model.right_arm_inverse_kinematics(self.right_arm_target_cart_pose, self.right_arm_target_cart_position, self.joint_target_position[7:14])

                    if (self.Kinematic_Model.right_arm_inverse_kinematics_solution_success_flag):
                        ## 导纳控制需要的参数赋值
                        self.right_arm_effector_pre_acc = self.right_arm_effector_current_acc
                        self.right_arm_effector_pre_speed = self.right_arm_effector_current_speed
                        self.joint_target_position[7:14] = self.Kinematic_Model.right_arm_interpolation_result
                    else:
                        self.joint_target_position[7:14] = self.last_joint_target_position[7:14]

                else:
                    self.joint_target_position[7:14] = self.last_joint_target_position[7:14]
            else:
                ## 拖动示教使用雅可比矩阵方案

                # 左臂的拖动处理
                Ftmp = math.sqrt(self.force_control_data.left_arm_FT_original_MAF_compensation[0] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation[1] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation[2] ** 2) 
                Mtmp = math.sqrt(self.force_control_data.left_arm_FT_original_MAF_compensation[3] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation[4] ** 2 + self.force_control_data.left_arm_FT_original_MAF_compensation[5] ** 2)
                # print("请注意进入拖动状态 Ftmp = {}".format(Ftmp))
                # print("请注意进入拖动状态 Mtmp = {}".format(Mtmp))                

                # 这个参数 需要进一步调节 直到拖动时感觉不出明显的卡顿为止
                if (Ftmp > 2) or (Mtmp > 1):
                    # 直接使用外力 求末端各个方向对应的速度值 后雅可比矩阵映射到各个关节对应的速度 再速度积分求位置 self.FT_original_MAF_compensation是传感器坐标系下补偿后的力传感器数据 
                    # 导纳控制输出笛卡尔空间下的速度
                    self.left_arm_effector_current_acc = (self.force_control_data.left_arm_FT_original_MAF_compensation - self.left_arm_admittance_control_B @ self.left_arm_effector_pre_speed) / self.left_arm_admittance_control_M 
                    self.left_arm_effector_current_speed = (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc) / 2 * (self.interpolation_period / 1000)


                    # 获取雅可比矩阵的函数接口 
                    Jacobians = self.Kinematic_Model.left_arm_Jacobians(self.joint_target_position[:7])

                    # 计算伪逆矩阵
                    Jacobians_inv = np.linalg.pinv(Jacobians)
                    self.left_arm_joint_target_speed = np.array(Jacobians_inv @ self.left_arm_effector_current_speed)
                    self.left_arm_joint_target_speed = (0.01 * self.left_arm_joint_target_speed + 0.99 * self.left_arm_joint_pre_speed)

                    self.left_arm_effector_pre_acc = self.left_arm_effector_current_acc
                    self.left_arm_effector_pre_speed = self.left_arm_effector_current_speed
                    self.left_arm_joint_pre_speed = self.left_arm_joint_target_speed

                    self.joint_target_position[:7] = self.joint_target_position[:7] + self.left_arm_joint_target_speed * (self.interpolation_period / 1000)

                else:
                    self.joint_target_position[:7] = self.joint_target_position[:7] 

                
                # 右臂的拖动处理
                Ftmp_right = math.sqrt(self.force_control_data.right_arm_FT_original_MAF_compensation[0] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation[1] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation[2] ** 2) 
                Mtmp_right = math.sqrt(self.force_control_data.right_arm_FT_original_MAF_compensation[3] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation[4] ** 2 + self.force_control_data.right_arm_FT_original_MAF_compensation[5] ** 2)
                # print("请注意进入拖动状态 Ftmp_right = {}".format(Ftmp_right))
                # print("请注意进入拖动状态 Ftmp_right = {}".format(Ftmp_right))

                # 这个参数 需要进一步调节 直到拖动时感觉不出明显的卡顿为止
                if (Ftmp_right > 2) or (Mtmp_right > 1):
                    # 直接使用外力 求末端各个方向对应的速度值 后雅可比矩阵映射到各个关节对应的速度 再速度积分求位置 self.FT_original_MAF_compensation是传感器坐标系下补偿后的力传感器数据 
                    # 导纳控制输出笛卡尔空间下的速度
                    self.right_arm_effector_current_acc = (self.force_control_data.right_arm_FT_original_MAF_compensation - self.right_arm_admittance_control_B @ self.right_arm_effector_pre_speed) / self.right_arm_admittance_control_M 
                    self.right_arm_effector_current_speed = (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc) / 2 * (self.interpolation_period / 1000)

                    # 获取雅可比矩阵的函数接口 
                    Jacobians = self.Kinematic_Model.right_arm_Jacobians(self.joint_target_position[7:14])

                    # 计算伪逆矩阵
                    Jacobians_inv = np.linalg.pinv(Jacobians)
                    self.right_arm_joint_target_speed = np.array(Jacobians_inv @ self.right_arm_effector_current_speed)
                    self.right_arm_joint_target_speed = (0.01 * self.right_arm_joint_target_speed + 0.99 * self.right_arm_joint_pre_speed)

                    self.right_arm_effector_pre_acc = self.right_arm_effector_current_acc
                    self.right_arm_effector_pre_speed = self.right_arm_effector_current_speed
                    self.right_arm_joint_pre_speed = self.right_arm_joint_target_speed

                    self.joint_target_position[7:14] = self.joint_target_position[7:14] + self.right_arm_joint_target_speed * (self.interpolation_period / 1000)

                else:
                    self.joint_target_position[7:14] = self.joint_target_position[7:14]

            if self.whether_save_drag_position == True:
                with open("joint_target_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.joint_target_position)           

            self.lcm_handler.upper_body_data_publisher(self.joint_target_position)    
            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 2毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间


