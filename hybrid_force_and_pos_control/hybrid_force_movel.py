import numpy as np
from lcm_handler import LCM_Handler
from robot_kinematics_and_dynamics_models.Kinematic_Model import Kinematic_Model
from dynamics_related_functions.zero_force_drag import Zero_Force_Drag
from dynamics_related_functions.collision_detection import Collision_Detection
import time
import math
from trajectory_plan.seven_segment_speed_plan import seven_segment_speed_plan
from copy import deepcopy
import quaternion  # 这将 numpy 的 ndarray 类型扩展以支持四元数
import pinocchio as pin
import sys
import csv



class Hybrid_Force_MoveL:
    def __init__(self, lcm_hancle, collision_detection, force_control_data_cal):
        self.lcm_hancle = lcm_hancle
        self.collision_detection = collision_detection
        self.force_control_data = force_control_data_cal
        self.kinematic_model = Kinematic_Model()
        # 绑定机器人urdf模型

        # 设置movel速度规划部分的参数
        self.movel_plan_jerk_max = 0.75
        self.movel_plan_acc_max = 0.5
        self.movel_plan_speed_max = 0.2

        # 此处与movel规划的参数意义不同，这里表示经过计算后的与力垂直的轨迹
        self.movel_plan_current_cart_position = None
        self.movel_plan_current_cart_pose = None
        self.movel_plan_current_cart_quat = None
        self.movel_plan_target_cart_position = None
        self.movel_plan_target_cart_pose = None
        self.movel_plan_target_cart_quat = None
        self.movel_plan_position_delta_disp = np.zeros(3)
        self.movel_plan_displacement = 0       
        self.movel_plan_target_position_list = None
        self.movel_plan_current_joint_position = None
        self.movel_plan_target_joint_position = None

        # 右臂速度规划参数
        self.right_arm_movel_plan_current_cart_position = None
        self.right_arm_movel_plan_current_cart_pose = None
        self.right_arm_movel_plan_current_cart_quat = None
        self.right_arm_movel_plan_target_cart_position = None
        self.right_arm_movel_plan_target_cart_pose = None
        self.right_arm_movel_plan_target_cart_quat = None
        self.right_arm_movel_plan_position_delta_disp = np.zeros(3)
        self.right_arm_movel_plan_displacement = 0       
        self.right_arm_movel_plan_target_position_list = None
        self.right_arm_movel_plan_current_joint_position = None
        self.right_arm_movel_plan_target_joint_position = None
        self.cart_interpolation_position = np.zeros(3)
        self.cart_interpolation_pose = np.zeros((3, 3))
        self.right_arm_cart_interpolation_position = np.zeros(3)
        self.right_arm_cart_interpolation_pose = np.zeros((3, 3))

        self.whether_save_movel_position = 0
        self.left_arm_inverse_kinematics_solution_success_flag = None
        self.right_arm_inverse_kinematics_solution_success_flag = None

        self.speed_plan = None
        self.MIN_VAL = 0.0000001 
        self.interpolation_period = 2
        self.interpolation_result = None

        # SE3表示的左右臂笛卡尔空间位姿
        self.left_arm_interp = pin.SE3.Identity()
        self.right_arm_interp = pin.SE3.Identity()

        self.right_arm_movel_plan_current_cart = pin.SE3.Identity()
        self.right_arm_movel_plan_target_cart = pin.SE3.Identity()
        self.movel_plan_current_cart = pin.SE3.Identity()
        self.movel_plan_target_cart = pin.SE3.Identity()


        # 力控部分的参数
        # 机器人笛卡尔空间下的位置、速度、加速度
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

        # 恒力跟踪控制中的期望力设置   
        self.left_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])
        self.right_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])

        # 导纳控制参数设置（M,B，不含有旋转项控制）
        self.left_arm_admittance_control_M = 0.1
        self.left_arm_admittance_control_B = 0.05

        self.right_arm_admittance_control_M = 0.1
        self.right_arm_admittance_control_B = 0.05

    def cal_hybrid_force_movel_plan_data_by_joint(self, current_position, target_position, target_FT_data):
        """
        输入：左右臂当前与目标的关节,左右臂末端目标力数据
        对左右臂速度规划参数进行计算
        """
        self.left_arm_target_FT_data = target_FT_data[:6]
        self.right_arm_target_FT_data = target_FT_data[6:12]

        self.movel_plan_current_cart = deepcopy(self.kinematic_model.left_arm_forward_kinematics(current_position[:7]))
        self.right_arm_movel_plan_current_cart = deepcopy(self.kinematic_model.right_arm_forward_kinematics(current_position[7:14]))

        self.movel_plan_target_cart = deepcopy(self.kinematic_model.right_arm_forward_kinematics(target_position[:7]))    # 计算左臂当前笛卡尔空间位置
        self.right_arm_movel_plan_target_cart = deepcopy(self.kinematic_model.right_arm_forward_kinematics(target_position[7:14]))    # 计算右臂当前笛卡尔空间位置

        self.cal_hybrid_force_movel_plan_data()



    def cal_hybrid_force_movel_plan_data_by_cart(self, left_arm_current_cart, right_arm_current_cart, left_arm_target_cart, right_arm_target_cart, target_FT_data):
        """
        输入：左右臂当前与目标的笛卡尔空间位姿,左右臂末端目标力数据
        对左右臂速度规划参数进行计算
        """        
        self.left_arm_target_FT_data = target_FT_data[:6]
        self.right_arm_target_FT_data = target_FT_data[6:12]

        self.movel_plan_current_cart = deepcopy(left_arm_current_cart)
        self.right_arm_movel_plan_current_cart = deepcopy(right_arm_current_cart)

        self.movel_plan_target_cart = deepcopy(left_arm_target_cart)
        self.right_arm_movel_plan_target_cart = deepcopy(right_arm_target_cart)

        self.cal_hybrid_force_movel_plan_data()




    def cal_hybrid_force_movel_plan_data(self):
        """
        计算hybrid_force_movel速度规划参数
        """

        # 左臂部分规划数据设置

        # self.left_arm_target_FT_data = target_FT_data[:6]
        self.movel_plan_current_cart_position = deepcopy(self.movel_plan_current_cart.position)    # 计算左臂当前笛卡尔空间位置
        self.movel_plan_current_cart_pose = deepcopy(self.movel_plan_current_cart.pose)    # 计算左臂当前笛卡尔空间姿态
        self.movel_plan_current_cart_quat = quaternion.from_rotation_matrix(self.movel_plan_current_cart)    # 计算左臂当前四元数姿态，采用四元数插值

        if self.whether_save_movel_position:
            print("self.movel_plan_current_cart_position  = {} ".format(self.movel_plan_current_cart_position))
            print("self.movel_plan_current_cart_pose  = {} ".format(self.movel_plan_current_cart_pose))

        self.movel_plan_target_cart_position = deepcopy(self.movel_plan_target_cart.position)    # 计算左臂目标笛卡尔空间位置
        self.movel_plan_target_cart_pose = deepcopy(self.movel_plan_target_cart.pose)    # 计算左臂目标笛卡尔空间姿态
        self.movel_plan_target_cart_quat = quaternion.from_rotation_matrix(self.movel_plan_target_cart_pose)    

        if self.whether_save_movel_position:
            print("self.movel_plan_target_cart_position  = {} ".format(self.movel_plan_target_cart_position))
            print("self.movel_plan_target_cart_pose  = {} ".format(self.movel_plan_target_cart_pose)) 

        # 求解实际的法向路径规划数据
        target_delta_dis = self.movel_plan_target_cart_position - self.movel_plan_current_cart_position
        force_direction = self.left_arm_target_FT_data[:3]
        if np.linalg.norm(force_direction) < self.MIN_VAL:
            print("左臂输入力过小，将执行movel指令")
            self.movel_plan_position_delta_disp = target_delta_dis
            self.movel_plan_displacement = np.linalg.norm(self.movel_plan_position_delta_disp)
        else:
            self.movel_plan_position_delta_disp = target_delta_dis - np.dot(target_delta_dis, force_direction) / np.dot(force_direction, force_direction) * force_direction
            self.movel_plan_target_cart_position = self.movel_plan_current_cart_position + self.movel_plan_position_delta_disp
            self.movel_plan_displacement = np.linalg.norm(self.movel_plan_position_delta_disp)
        
        
        if np.dot(self.movel_plan_current_cart_quat.components, self.movel_plan_target_cart_quat.components) < 0:
                self.movel_plan_target_cart_quat = - self.movel_plan_target_cart_quat   
    

        # 右臂部分规划数据设置
        # self.right_arm_target_FT_data = target_FT_data[6:12]

        self.right_arm_movel_plan_current_cart_position = deepcopy(self.right_arm_movel_plan_current_cart.position)    # 计算左臂当前笛卡尔空间位置
        self.right_arm_movel_plan_current_cart_pose = deepcopy(self.right_arm_movel_plan_current_cart.pose)    # 计算左臂当前笛卡尔空间姿态
        self.right_arm_movel_plan_current_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movel_plan_target_cart_pose)    # 计算左臂当前四元数姿态，采用四元数插值

        if self.whether_save_movel_position:
            print("self.movel_plan_current_cart_position  = {} ".format(self.right_arm_movel_plan_current_cart_position))
            print("self.movel_plan_current_cart_pose  = {} ".format(self.right_arm_movel_plan_current_cart_pose))

        self.right_arm_movel_plan_target_cart_position = deepcopy(self.right_arm_movel_plan_target_cart.position)    # 计算左臂目标笛卡尔空间位置
        self.right_arm_movel_plan_target_cart_pose = deepcopy(self.right_arm_movel_plan_target_cart.pose)    # 计算左臂目标笛卡尔空间姿态
        self.right_arm_movel_plan_target_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movel_plan_target_cart_pose)    

        if self.whether_save_movel_position: 
            print("self.right_arm_movel_plan_target_cart_position  = {} ".format(self.right_arm_movel_plan_target_cart_position))
            print("self.right_arm_movel_plan_target_cart_pose  = {} ".format(self.right_arm_movel_plan_target_cart_pose))

        # 求解实际的法向路径规划数据
        target_delta_dis = self.right_arm_movel_plan_target_cart_position - self.right_arm_movel_plan_current_cart_position
        force_direction = self.right_arm_target_FT_data[:3]
        self.right_arm_movel_plan_position_delta_disp = target_delta_dis - np.dot(target_delta_dis, force_direction) / np.dot(force_direction, force_direction) * force_direction
        
        if np.linalg.norm(force_direction) < self.MIN_VAL:
            print("右臂输入力过小，将执行movel指令")
            self.right_arm_movel_plan_position_delta_disp = target_delta_dis
            self.right_arm_movel_plan_displacement = np.linalg.norm(self.right_arm_movel_plan_position_delta_disp)
        else:
            self.right_arm_movel_plan_position_delta_disp = target_delta_dis - np.dot(target_delta_dis, force_direction) / np.dot(force_direction, force_direction) * force_direction
            self.right_arm_movel_plan_target_cart_position = self.right_arm_movel_plan_current_joint_position + self.right_arm_movel_plan_position_delta_disp
            self.right_arm_movel_plan_displacement = np.linalg.norm(self.right_arm_movel_plan_current_joint_position)

        if np.dot(self.right_arm_movel_plan_current_cart_quat.components, self.right_arm_movel_plan_target_cart_quat.components) < 0:
                self.right_arm_movel_plan_target_cart_quat = - self.right_arm_movel_plan_target_cart_quat        


    def hybrid_force_movel_plan_interpolation(self):
        """
        执行力位混合控制的参数计算与直线速度规划后，执行此函数进行插值和数据的实时发送
        """
        self.Collision_Detection.start_collision_detection()

        for interpolation_time in np.range(0, self.speed_plan.time_length, self.interpolation_period / 1000):
            # 记录循环开始的时间
            start_time = time.time()
            if 0 <= interpolation_time <= self.speed_plan.accacc_time:
                self.speed_plan.cal_accacc_segment_data(interpolation_time)
            elif self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.uniacc_time + self.speed_plan.accacc_time:
                interpolation_time = interpolation_time - self.speed_plan.accacc_time
                self.speed_plan.cal_uniacc_segment_data(interpolation_time)
            elif self.speed_plan.uniacc_time + self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.acceleration_segment_time:
                interpolation_time = interpolation_time - (self.speed_plan.uniacc_time + self.speed_plan.accacc_time)
                self.speed_plan.cal_decacc_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time:
                interpolation_time = interpolation_time - self.speed_plan.acceleration_segment_time
                self.speed_plan.cal_unispeed_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time:
                interpolation_time = interpolation_time - (self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time)
                self.speed_plan.cal_accdec_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time < interpolation_time <= self.speed_plan.time_length - self.speed_plan.decdec_time:
                interpolation_time = interpolation_time - (self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time)
                self.speed_plan.cal_unidec_segment_data(interpolation_time)
            else:
                interpolation_time = interpolation_time - (self.speed_plan.time_length - self.speed_plan.decdec_time)
                self.speed_plan.cal_decdec_segment_data(interpolation_time)

            
            # 计算左臂插补位置与姿态
            # 轨迹规划部分分量
            cart_position_increment_1 = self.speed_plan.cur_disp_normalization_ratio * self.movel_plan_position_delta_disp

            # 恒力控制部分分量
            # 读取当前末端力数据
            current_FT = self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[:3]
            target_FT = self.left_arm_target_FT_data[:3]   # 只提取轴向目标力分量
            err_F_num = np.dot(target_FT - current_FT, target_FT) / np.linalg.norm(target_FT)


            # 采用导纳控制计算末端位移（将三分力看作一维，避免对轨迹规划部分产生影响）
            self.left_arm_effector_current_acc = (err_F_num - self.left_arm_admittance_control_B * np.linalg.norm(self.left_arm_effector_pre_acc)) / self.left_arm_admittance_control_M
            self.left_arm_effector_current_acc = 0.5 * (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc)

            self.left_arm_effector_current_speed = (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc) * (self.interpolation_period / 1000)
            self.left_arm_effector_current_speed = 0.5 * self.left_arm_effector_current_speed + 0.5 * self.left_arm_effector_pre_speed

            # 将导纳部分的位移增量映射到力的方向
            if np.linalg.norm(target_FT) > self.MIN_VAL:
                cart_position_increment_2 = self.left_arm_effector_current_speed * (self.interpolation_period / 1000) * target_FT / np.linalg.norm(target_FT)
            else:
                cart_position_increment_2 = np.zeros(3)
            self.cart_interpolation_position = self.movel_plan_current_cart_position + cart_position_increment_1 + cart_position_increment_2
            
            # 四元数球面线性插值
            slerped_quaternions = quaternion.slerp(self.movel_plan_current_cart_quat, self.movel_plan_target_cart_quat, 0, 1, self.speed_plan.cur_disp_normalization_ratio)
            self.cart_interpolation_pose = quaternion.as_rotation_matrix(slerped_quaternions)


            # self.left_arm_interp.translation = self.cart_interpolation_position
            # self.left_arm_interp.rotation = self.cart_interpolation_pose

            #  左臂逆解 逆解 逆解 
            self.Kinematic_Model.left_arm_inverse_kinematics(self.cart_interpolation_pose, self.cart_interpolation_position, self.movel_plan_current_joint_position[:7])

            
            # 计算右臂插补位置与姿态

            # 轨迹规划部分分量
            cart_position_increment_1 = self.speed_plan.cur_disp_normalization_ratio * self.right_arm_movel_plan_position_delta_disp
            
            # 读取当前末端力数据
            current_FT = self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[:3]
            target_FT = self.right_arm_target_FT_data[:3]   # 只提取轴向目标力分量
            err_F_num = np.dot(target_FT - current_FT, target_FT) / np.linalg.norm(target_FT)

            # 采用导纳控制计算末端位移（将三分力看作一维，避免对轨迹规划部分产生影响）
            self.right_arm_effector_current_acc = (err_F_num - self.right_arm_admittance_control_B * np.linalg.norm(self.right_arm_effector_pre_acc)) / self.right_arm_admittance_control_M
            self.right_arm_effector_current_acc = 0.5 * (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc)

            self.right_arm_effector_current_speed = (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc) * (self.interpolation_period / 1000)
            self.right_arm_effector_current_speed = 0.5 * self.right_arm_effector_current_speed + 0.5 * self.right_arm_effector_pre_speed


            # 将导纳部分的位移增量映射到力的方向
            if  np.linalg.norm(target_FT) > self.MIN_VAL:
                cart_position_increment_2 = self.right_arm_effector_current_speed * (self.interpolation_period / 1000) * target_FT / np.linalg.norm(target_FT)
            else: 
                cart_position_increment_2 = np.zeros(3)


            self.right_arm_cart_interpolation_position = self.right_arm_movel_plan_current_cart_position + cart_position_increment_1 + cart_position_increment_2

            
            slerped_quaternions = quaternion.slerp(self.right_arm_movel_plan_current_cart_quat, self.right_arm_movel_plan_target_cart_quat, 0, 1, self.speed_plan.cur_disp_normalization_ratio)
            self.right_arm_cart_interpolation_pose = quaternion.as_rotation_matrix(slerped_quaternions)

            self.right_arm_interp.translation = self.right_arm_cart_interpolation_position
            self.right_arm_interp.rotation = self.right_arm_cart_interpolation_pose

            # # 右臂逆解 逆解 逆解
            self.Kinematic_Model.right_arm_inverse_kinematics(self.right_arm_interp.rotation, self.right_arm_interp.translation, self.movel_plan_current_joint_position[7:14])


            # 校验逆解是否成功
            if (self.Kinematic_Model.left_arm_inverse_kinematics_solution_success_flag and self.Kinematic_Model.right_arm_inverse_kinematics_solution_success_flag) == False:
                print("逆解失败咯，机器人应该停止运行了，请调整合理的位置呀！！！！")
                self.interpolation_result = self.movel_plan_current_joint_position
                break
            else:
                self.interpolation_result = self.movel_plan_current_joint_position
                self.interpolation_result[7:14] = self.Kinematic_Model.right_arm_interpolation_result
                self.interpolation_result[:7] = self.Kinematic_Model.left_arm_interpolation_result
            

            # 保存运动数据
            if self.whether_save_movel_position:
                self.interpolation_result_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.interpolation_result[:7]))
                self.interpolation_result_cart_position = self.interpolation_result_cart.translation
                self.interpolation_result_cart_pose = self.interpolation_result_cart.rotation
                self.interpolation_result_cart_quat = quaternion.from_rotation_matrix(self.interpolation_result_cart_pose)

                with open("movel_interpolate_trajectory.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result)

                with open("movel_left_arm_interpolation_result_cart_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result_cart_position)

                with open("movel_left_arm_interpolation_result_cart_pose.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result_cart_pose)

                self.right_arm_interpolation_result_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.interpolation_result[7:14]))
                self.right_arm_interpolation_result_cart_position = self.right_arm_interpolation_result_cart.translation
                self.right_arm_interpolation_result_cart_pose = self.right_arm_interpolation_result_cart.rotation
                self.right_arm_interpolation_result_cart_quat = quaternion.from_rotation_matrix(self.right_arm_interpolation_result_cart_pose)


                with open("movel_right_arm_interpolation_result_cart_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.right_arm_interpolation_result_cart_position)

                with open("movel_right_arm_interpolation_result_cart_pose.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.right_arm_interpolation_result_cart_pose)


            # 碰撞检测
            if(self.Collision_Detection.collision_detection_index):
                print("发生了碰撞，结束碰撞检测线程，退出当前插补函数！！！！")
                self.Collision_Detection.stop_collision_detection()
                sys.exit()    # 退出程序循环，机械臂停止运动


            # 数据发布
            # 双臂关节数据
            self.lcm_handler.upper_body_data_publisher(self.interpolation_result)            
            self.movel_plan_current_joint_position = self.interpolation_result


            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间


        print("运行结束，到达目标点位！！！")
        self.Collision_Detection.stop_collision_detection()   


    def robot_hybrid_force_movel_control_by_cart(self, left_arm_current_position, left_arm_target_position, right_arm_current_position, right_arm_target_position,
                                   left_arm_target_FT_data, right_arm_target_FT_data):
        """
        基于笛卡尔位姿输入的双臂混合力控制+位置控制
        """
        
        # self.movel_plan_current_joint_position = robot_current_qpos  
        self.cal_hybrid_force_movel_plan_data_by_cart(left_arm_current_position, left_arm_target_position, right_arm_current_position, right_arm_target_position, left_arm_target_FT_data, right_arm_target_FT_data)  
        self.speed_plan = seven_segment_speed_plan(self.movel_plan_jerk_max, self.movel_plan_acc_max, self.movel_plan_speed_max, max(self.movel_plan_displacement, self.right_arm_movel_plan_displacement))  
        self.hybrid_force_movel_plan_interpolation()


    def robot_hybrid_force_movel_by_joint(self, current_position, target_position, target_FT_data):
        """
        基于关节输入的双臂混合力控制+位置控制
        """
        self.cal_hybrid_force_movel_plan_data_by_joint(current_position, target_position, target_FT_data)
        self.speed_plan = seven_segment_speed_plan(self.movel_plan_jerk_max, self.movel_plan_acc_max, self.movel_plan_speed_max, max(self.movel_plan_displacement, self.right_arm_movel_plan_displacement))
        self.hybrid_force_movel_plan_interpolation()