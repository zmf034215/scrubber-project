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
    def __init__(self, LCMHandler, force_control_data_cal, Kinematic_Model):
        # lcm
        self.lcm_handler = LCMHandler
        self.force_control_data = force_control_data_cal
        self.Kinematic_Model = Kinematic_Model

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

        self.left_arm_joint_pre_speed = np.array([0, 0, 0, 0, 0, 0, 0])
        self.right_arm_joint_pre_speed = np.array([0, 0, 0, 0, 0, 0, 0])

        # 是否需要使用逆解方案来实现拖动以及恒力跟踪标志位 0不使用 1使用
        self.force_sensor_drag_teach_whether_use_IK = True

        # 是否存储拖动/恒力跟踪产生的期望位置曲线标志位 0 不保存 1 保存
        self.whether_save_drag_position = False
        self.whether_save_constant_force_track_control_position = False

        # 恒力跟踪控制中的期望力设置   
        self.left_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])
        self.right_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])


        ## 导纳控制实现拖动示教的系数设置 
        # 雅可比方案的导纳控制参数设置  
        self.left_arm_admittance_control_M = np.array([0.1, 0.1, 0.1, 100, 100, 100])
        self.left_arm_admittance_control_B = np.array([0.05, 0.05, 0.05, 100, 100, 100])

        self.right_arm_admittance_control_M = np.array([0.1, 0.1, 0.1, 100, 100, 100])
        self.right_arm_admittance_control_B = np.array([0.05, 0.05, 0.05, 100, 100, 100])

        # 逆解方案的导纳控制参数设置
        self.left_arm_admittance_control_M_end_cartesian_space_plan = np.array([0.01, 0.01, 0.01, 10, 10, 10])
        self.left_arm_admittance_control_B_end_cartesian_space_plan = np.array([0.1, 0.1, 0.1, 5, 5, 5])

        self.right_arm_admittance_control_M_end_cartesian_space_plan = np.array([0.01, 0.01, 0.01, 10, 10, 10])
        self.right_arm_admittance_control_B_end_cartesian_space_plan = np.array([0.1, 0.1, 0.1, 5, 5, 5])


        ## 导纳控制实现恒力跟踪的参数设置
        # 逆解方案的导纳控制参数设置
        self.left_arm_admittance_control_M_end_cartesian_space_plan_force_tracking_control = np.array([0.1, 0.1, 0.1, 10, 10, 10])
        self.left_arm_admittance_control_B_end_cartesian_space_plan_force_tracking_control = np.array([0.5, 0.5, 0.5, 5, 5, 5])

        self.right_arm_admittance_control_M_end_cartesian_space_plan_force_tracking_control = np.array([0.1, 0.1, 0.1, 10, 10, 10])
        self.right_arm_admittance_control_B_end_cartesian_space_plan_force_tracking_control = np.array([0.5, 0.5, 0.5, 5, 5, 5])

        self.interpolation_period = 2
        self.joint_target_position = None



        # 锁轴拖动 方向标志位 纯笛卡尔/逆解的方案中有锁轴拖动功能
        self.left_arm_force_sensor_drag_teach_lock_axis_sign = np.array([1, 1, 1, 1, 1, 1])
        self.right_arm_force_sensor_drag_teach_lock_axis_sign = np.array([1, 1, 1, 1, 1, 1])


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
                if (Ftmp > 3) or (Mtmp > 1.5):
                    self.force_control_left_arm_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.joint_target_position[:7]))
                    self.left_arm_target_cart_position = deepcopy(self.force_control_left_arm_current_cart.translation)
                    self.left_arm_target_cart_pose = deepcopy(self.force_control_left_arm_current_cart.rotation)                    
                    self.left_arm_effector_pre_position = self.left_arm_target_cart_position

                    # 导纳控制输出笛卡尔空间下的速度
                    self.left_arm_effector_current_acc = (self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system - self.left_arm_admittance_control_B_end_cartesian_space_plan @ self.left_arm_effector_pre_speed) / self.left_arm_admittance_control_M_end_cartesian_space_plan 
                    self.left_arm_effector_current_acc = 0.5 * self.left_arm_effector_current_acc + 0.5 * self.left_arm_effector_pre_acc

                    self.left_arm_effector_current_speed = (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc) * (self.interpolation_period / 1000)
                    self.left_arm_effector_current_speed = 0.5 * self.left_arm_effector_current_speed + 0.5 * self.left_arm_effector_pre_speed

                    ## 纯笛卡尔/逆解的方案中有锁轴拖动功能
                    self.left_arm_effector_current_speed = self.left_arm_effector_current_speed * self.left_arm_force_sensor_drag_teach_lock_axis_sign


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

                if (Ftmp_right > 3) or (Mtmp_right > 1.5):
                    self.force_control_right_arm_current_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.joint_target_position[7:14]))
                    self.right_arm_target_cart_position = deepcopy(self.force_control_right_arm_current_cart.translation)
                    self.right_arm_target_cart_pose = deepcopy(self.force_control_right_arm_current_cart.rotation)                    
                    self.right_arm_effector_pre_position = self.right_arm_target_cart_position


                    # 导纳控制输出笛卡尔空间下的速度
                    self.right_arm_effector_current_acc = (self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system - self.right_arm_admittance_control_B_end_cartesian_space_plan @ self.right_arm_effector_pre_speed) / self.right_arm_admittance_control_M_end_cartesian_space_plan 
                    self.right_arm_effector_current_acc = (0.5 * self.right_arm_effector_current_acc + 0.5 * self.right_arm_effector_pre_acc)

                    self.right_arm_effector_current_speed = (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc) * (self.interpolation_period / 1000)
                    self.right_arm_effector_current_speed = 0.5 * self.right_arm_effector_current_speed + 0.5 * self.right_arm_effector_pre_speed

                    ## 纯笛卡尔/逆解的方案中有锁轴拖动功能
                    self.right_arm_effector_current_speed = self.right_arm_effector_current_speed * self.right_arm_force_sensor_drag_teach_lock_axis_sign

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
                if (Ftmp > 3) or (Mtmp > 1.5):
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
                if (Ftmp_right > 3) or (Mtmp_right > 1.5):
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


    def constant_force_tracking_control(self):
        self.joint_target_position = self.lcm_handler.joint_current_pos
        self.last_joint_target_position = self.joint_target_position

        # 期望力不为0的方向 产生对应的速度值 期望力为0 不产生
        left_arm_target_FT_data_index = np.zeros_like(self.left_arm_target_FT_data)
        left_arm_target_FT_data_index = np.where(self.left_arm_target_FT_data != 0, 1, 0)


        # 期望力不为0的方向 产生对应的速度值 期望力为0 不产生
        right_arm_target_FT_data_index = np.zeros_like(self.right_arm_target_FT_data)
        right_arm_target_FT_data_index = np.where(self.right_arm_target_FT_data != 0, 1, 0)



        while(1):
            start_time = time.time()  # 记录循环开始的时间

            self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system = self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system * left_arm_target_FT_data_index
            FT_data_err = self.left_arm_target_FT_data - self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system
            Ftmp = math.sqrt(FT_data_err[0] ** 2 + FT_data_err[1] ** 2 + FT_data_err[2] ** 2) 
            Mtmp = math.sqrt(FT_data_err[3] ** 2 + FT_data_err[4] ** 2 + FT_data_err[5] ** 2)
            # print("请注意进入拖动状态 Ftmp = {}".format(Ftmp))
            # print("请注意进入拖动状态 Mtmp = {}".format(Mtmp))

            if (Ftmp > 0.1) or (Mtmp > 0.02):
                # 正运动学 计算末端位置以及姿态 
                self.force_control_left_arm_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.joint_target_position[:7]))
                self.left_arm_target_cart_position = deepcopy(self.force_control_left_arm_current_cart.translation)
                self.left_arm_target_cart_pose = deepcopy(self.force_control_left_arm_current_cart.rotation)                    
                self.left_arm_effector_pre_position = self.left_arm_target_cart_position

                # 导纳控制输出笛卡尔空间下的速度
                self.left_arm_effector_current_acc = (FT_data_err - self.left_arm_admittance_control_B_end_cartesian_space_plan_force_tracking_control @ self.left_arm_effector_pre_speed) / self.left_arm_admittance_control_M_end_cartesian_space_plan_force_tracking_control
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

            

            # 右臂恒力跟踪的处理
            self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system = self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system * right_arm_target_FT_data_index
            FT_data_err = self.right_arm_target_FT_data - self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system
            Ftmp = math.sqrt(FT_data_err[0] ** 2 + FT_data_err[1] ** 2 + FT_data_err[2] ** 2) 
            Mtmp = math.sqrt(FT_data_err[3] ** 2 + FT_data_err[4] ** 2 + FT_data_err[5] ** 2)
            # print("请注意进入拖动状态 Ftmp = {}".format(Ftmp))
            # print("请注意进入拖动状态 Mtmp = {}".format(Mtmp))

            if (Ftmp > 0.1) or (Mtmp > 0.02):
                # 正运动学 计算末端位置以及姿态 
                self.force_control_right_arm_current_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.joint_target_position[7:14]))
                self.right_arm_target_cart_position = deepcopy(self.force_control_right_arm_current_cart.translation)
                self.right_arm_target_cart_pose = deepcopy(self.force_control_right_arm_current_cart.rotation)                    
                self.right_arm_effector_pre_position = self.right_arm_target_cart_position


                # 导纳控制输出笛卡尔空间下的速度
                self.right_arm_effector_current_acc = (FT_data_err - self.right_arm_admittance_control_B_end_cartesian_space_plan_force_tracking_control @ self.right_arm_effector_pre_speed) / self.right_arm_admittance_control_M_end_cartesian_space_plan_force_tracking_control 
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


            if self.whether_save_constant_force_track_control_position == True:
                with open("joint_target_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.joint_target_position)           

            self.lcm_handler.upper_body_data_publisher(self.joint_target_position)    
            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 2毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间



    def move_down_until_force(self, arm='right', target_force=10.0, hold_time=0.5):
        """
        机器人沿Z向下压，直到六维力传感器Z方向力达到 target_force (单位N),
        稳定保持 hold_time 秒。
        """
        print("⬇️ 开始下压，直到 Z 方向接触力达到目标值...")
        # step_z = -0.001  # 每次下压1mm
        max_attempts = 50
        Kpz = 0.0005

        for _ in range(max_attempts):
            ft = self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system if arm == 'right' \
                else self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system

            if ft is None:
                continue

            current_force_z = ft[2]
            if abs(current_force_z) >= target_force:
                print(f"✅ 接触力已达 {current_force_z:.2f} N，开始保持 {hold_time}s...")
                time.sleep(hold_time)
                return

            error = target_force - abs(current_force_z)
            dz = np.clip(error * Kpz, -0.002, 0.002)
            delta = np.array([0.0, 0.0, dz])
            success = self.Kinematic_Model.move_relative(arm, delta)
            if not success :
                print("❌ 超出机械臂可达空间！！！")
                exit()
            time.sleep(self.interpolation_period / 1000.0)

        print("⚠️ 达不到目标力，停止下压，***程序终止***")
        exit()

    def desktop_wiping_force_tracking_control(self,arm='right',start_pose = None, hold_time = 0.5,wipe_direction=np.array([1.0, 0.0]), wipe_step=0.002, wipe_total_distance=0.3):
        """
        执行桌面擦拭任务：
        1. 运动到起始位姿；
        2. 沿Z方向下压，直到目标力（10N）；
        3. 保持一定时间；
        4. 沿XY方向擦拭，Z方向保持恒定力。
        """

        target_force_z = self.right_arm_target_FT_data[2] if arm == 'right' else self.left_arm_target_FT_data[2]
        
        # 先移动至起始位姿
        print("开始移动到起始位姿...")
        success = self.Kinematic_Model.move_to_start_pose(arm, start_pose)
        if not success:
            print("❌ 起始位姿运动失败，程序终止。")
            exit()
        else:
            print("✅ 已到达起始位姿。")
        # 等待运动完成或加反馈判断
        time.sleep(2)
        
        # 先下压至10N
        self.move_down_until_force(arm=arm,target_force=abs(target_force_z), hold_time=0.5)

        print("🧽 开始擦拭...")
        # wipe_steps = int(wipe_total_distance / wipe_step)
        # dx = wipe_direction[0] * wipe_step
        # dy = wipe_direction[1] * wipe_step
       

        # for i in range(wipe_steps):
        #     # 获取当前力数据
        #     ft = self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system if arm == 'right' \
        #          else self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system
            
        #     if ft is None:
        #         continue  # 力数据无效，跳过当前循环    

        #     current_force_z = ft[2]
            
        #     error_z = target_force_z - current_force_z
        #     dz = np.clip(error_z * 0.0005, -0.002, 0.002)
            
        #     delta = np.array([dx, dy, dz])
        #     success = self.Kinematic_Model.move_relative(arm, delta)
        #     if not success :
        #         print("❌ 超出机械臂可达空间！！！")
        #         exit()

        #     time.sleep(self.interpolation_period / 1000.0)

        dx = wipe_direction[0] * wipe_total_distance
        dy = wipe_direction[1] * wipe_total_distance
        dz = 0.0
        delta = np.array([dx, dy, dz])
        success = self.Kinematic_Model.move_relative_FT(arm, delta, target_force_z)
        print("ca shi****")
        if not success :
            print("❌ 超出机械臂可达空间！！！")
            exit()
                
        print(">>> 全部完成，抬升 2 cm")
        time.sleep(2)
        self.Kinematic_Model.move_relative(arm, np.array([0, 0, 0.02]))
        self.Kinematic_Model.back_to_start_pose(arm,start_pose) 
        print("✅ 擦拭任务完成。")

