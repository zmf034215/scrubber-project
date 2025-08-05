import numpy as np
# from lcm_handler import LCM_Handler
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


class Hybrid_Force_MoveC:
    def __init__(self, lcm_hancle, collision_detection, force_control_data_cal):
        self.lcm_hancle = lcm_hancle
        self.collision_detection = collision_detection
        self.force_control_data = force_control_data_cal
        self.kinematic_model = Kinematic_Model()
        # 绑定机器人urdf模型
        
    # MOVEC 变量
        self.movec_plan_current_joint_position = None
        self.movec_plan_middle_joint_position = None
        self.movec_plan_target_joint_position = None

    # 左臂变量
        self.movec_plan_current_cart = pin.SE3.Identity()
        self.movec_plan_current_cart_position = None
        self.movec_plan_current_cart_pose = None
        self.movec_plan_current_cart_quat = None

        self.movec_plan_middle_cart = pin.SE3.Identity()
        self.movec_plan_middle_cart_position = None
        self.movec_plan_middle_cart_pose = None
        self.movec_plan_middle_cart_quat = None

        self.movec_plan_target_cart = pin.SE3.Identity()
        self.movec_plan_target_cart_position = None
        self.movec_plan_target_cart_pose = None
        self.movec_plan_target_cart_quat = None

        self.movec_plan_center = np.zeros(3)
        self.movec_plan_radius = None
        self.movec_plan_theta = None
        self.movec_plan_arc_length = None
        self.movec_plan_center_vec = None
        self.movec_plan_displacement = 0       
        self.movec_plan_target_position_list = None
        self.movec_plan_current_joint_position = None
        self.movec_plan_middle_joint_position = None
        self.movec_plan_target_joint_position = None

    # 右臂变量
        self.right_arm_movec_plan_current_cart = pin.SE3.Identity()
        self.right_arm_movec_plan_current_cart_position = None
        self.right_arm_movec_plan_current_cart_pose = None
        self.right_arm_movec_plan_current_cart_quat = None

        self.right_arm_movec_plan_middle_cart = pin.SE3.Identity()
        self.right_arm_movec_plan_middle_cart_position = None
        self.right_arm_movec_plan_middle_cart_pose = None
        self.right_arm_movec_plan_middle_cart_quat = None

        self.right_arm_movec_plan_target_cart = pin.SE3.Identity()
        self.right_arm_movec_plan_target_cart_position = None
        self.right_arm_movec_plan_target_cart_pose = None
        self.right_arm_movec_plan_target_cart_quat = None

        self.right_arm_movec_plan_center = np.zeros(3)
        self.right_arm_movec_plan_radius = None
        self.right_arm_movec_plan_theta = None
        self.right_arm_movec_plan_arc_length = None
        self.right_arm_movec_plan_center_vec = None
        self.right_arm_movec_plan_displacement = 0       
        self.right_arm_movec_plan_target_position_list = None
        self.right_arm_movec_plan_current_joint_position = None
        self.right_arm_movec_plan_middle_joint_position = None
        self.right_arm_movec_plan_target_joint_position = None

        self.movec_plan_jerk_max = 0.75
        self.movec_plan_acc_max = 0.5
        self.movec_plan_speed_max = 0.2 

        self.Kinematic_Model = Kinematic_Model()
        self.MIN_VAL = 0.0000001  
        self.interpolation_period = 2
        self.interpolation_result = None
        self.whether_save_movec_position = 0


        self.interpolation_result_cart = pin.SE3.Identity()
        self.interpolation_result_cart_position = None
        self.interpolation_result_cart_pose = None
        self.interpolation_result_cart_quat = None

        self.right_arm_interpolation_result_cart = pin.SE3.Identity()
        self.right_arm_interpolation_result_cart_position = None
        self.right_arm_interpolation_result_cart_pose = None
        self.right_arm_interpolation_result_cart_quat = None


        # 恒力跟踪控制中的期望力设置
        self.left_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])
        self.right_arm_target_FT_data = np.array([0, 0, 15, 0, 0, 0])

        # 导纳控制参数设置（M,B，不含有旋转项控制）
        self.left_arm_admittance_control_M = 0.1
        self.left_arm_admittance_control_B = 0.05

        self.right_arm_admittance_control_M = 0.1
        self.right_arm_admittance_control_B = 0.05

        # 左右臂末端笛卡尔位置，速度和加速度，用于阻抗/导纳控制
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

        

    def cal_hybrid_force_movec_plan_data_by_joint(self, current_position, middle_position, target_position,
                                        target_FT_data):
        # 输入的目标位置是关节指令形式
        self.movec_plan_current_joint_position = np.array(current_position)
        self.movec_plan_middle_joint_position = np.array(middle_position)
        self.movec_plan_target_joint_position = np.array(target_position)

        # 设置用于规划的笛卡尔空间位置（current,middle,target）
        self.movec_plan_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movec_plan_current_joint_position[:7]))
        self.right_arm_movec_plan_current_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movec_plan_current_joint_position[7:14]))

        self.movec_plan_middle_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movec_plan_middle_joint_position[:7]))
        self.right_arm_movec_plan_middle_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movec_plan_middle_joint_position[7:14]))

        self.movec_plan_target_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movec_plan_target_joint_position[:7]))
        self.right_arm_movec_plan_target_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movec_plan_target_joint_position[7:14]))


        self.cal_left_arm_movec_plan_data()
        self.cal_right_arm_movec_plan_data()

        self.left_arm_target_FT_data = target_FT_data[:6]
        self.right_arm_target_FT_data = target_FT_data[6:12]
        if np.linalg.norm(self.left_arm_target_FT_data) < self.MIN_VAL:
            print("左臂目标力为0，将执行moveC指令")

        if np.linalg.norm(self.right_arm_target_FT_data) < self.MIN_VAL:
            print("右臂目标力为0，将执行moveC指令")


    def cal_hybrid_force_movec_plan_data_by_cart(self, left_current_position, left_middle_position, left_target_position,
                                         right_current_position, right_middle_position, right_target_position,
                                         target_FT_data):
        # 输入的目标位置是笛卡尔指令形式
        # 设置用于规划的笛卡尔空间位置（current,middle,target）
        self.movec_plan_current_cart = deepcopy(left_current_position)
        self.right_arm_movec_plan_current_cart = deepcopy(right_current_position)

        self.movec_plan_middle_cart = deepcopy(left_middle_position)
        self.right_arm_movec_plan_middle_cart = deepcopy(right_middle_position)

        self.movec_plan_target_cart = deepcopy(left_target_position)
        self.right_arm_movec_plan_target_cart = deepcopy(right_target_position)


        self.cal_left_arm_movec_plan_data()
        self.cal_right_arm_movec_plan_data()

        self.left_arm_target_FT_data = target_FT_data[:6]
        self.right_arm_target_FT_data = target_FT_data[6:12]
        if np.linalg.norm(self.left_arm_target_FT_data) < self.MIN_VAL:
            print("左臂目标力为0，将执行moveC指令")

        if np.linalg.norm(self.right_arm_target_FT_data) < self.MIN_VAL:
            print("右臂目标力为0，将执行moveC指令")



    def cal_left_arm_movec_plan_data(self):
        # 获取左臂当前位置对应的末端笛卡尔位置姿态和四元数
        # self.movec_plan_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movec_plan_current_joint_position[:7]))
        self.movec_plan_current_cart_position = self.movec_plan_current_cart.translation
        self.movec_plan_current_cart_pose = self.movec_plan_current_cart.rotation
        self.movec_plan_current_cart_quat = quaternion.from_rotation_matrix(self.movec_plan_current_cart_pose)
        if self.whether_save_movec_position:
            print("self.movec_plan_current_cart_position  = {} ".format(self.movec_plan_current_cart_position))
            print("self.movec_plan_current_cart_pose  = {} ".format(self.movec_plan_current_cart_pose))

        # 获取左臂中间点位置对应的末端笛卡尔位置姿态和四元数
        # self.movec_plan_middle_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movec_plan_middle_joint_position[:7]))
        self.movec_plan_middle_cart_position = self.movec_plan_middle_cart.translation
        self.movec_plan_middle_cart_pose = self.movec_plan_middle_cart.rotation
        self.movec_plan_middle_cart_quat = quaternion.from_rotation_matrix(self.movec_plan_middle_cart_pose)
        if self.whether_save_movec_position:
            print("self.movec_plan_middle_cart_position  = {} ".format(self.movec_plan_middle_cart_position))
            print("self.movec_plan_middle_cart_pose  = {} ".format(self.movec_plan_middle_cart_pose))        

        # 获取左臂期望位置对应的末端笛卡尔位置姿态和四元数
        # self.movec_plan_target_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movec_plan_target_joint_position[:7]))
        self.movec_plan_target_cart_position = self.movec_plan_target_cart.translation
        self.movec_plan_target_cart_pose = self.movec_plan_target_cart.rotation
        self.movec_plan_target_cart_quat = quaternion.from_rotation_matrix(self.movec_plan_target_cart_pose)
        if self.whether_save_movec_position:
            print("self.movec_plan_target_cart_position  = {} ".format(self.movec_plan_target_cart_position))
            print("self.movec_plan_target_cart_pose  = {} ".format(self.movec_plan_target_cart_pose))


        if(np.array_equal(self.movec_plan_current_cart_position, self.movec_plan_middle_cart_position) or np.array_equal(self.movec_plan_middle_cart_position, self.movec_plan_target_cart_position) or np.array_equal(self.movec_plan_current_cart_position, self.movec_plan_target_cart_position)):
            raise ValueError("设置的三个点位不能存在两个点完全重合，请重新设置圆弧的中间点位和目标点位")
        
        # 计算圆弧运动中圆弧对应的圆心位置
        x1 = self.movec_plan_current_cart_position[0]
        y1 = self.movec_plan_current_cart_position[1]
        z1 = self.movec_plan_current_cart_position[2]

        x2 = self.movec_plan_middle_cart_position[0]
        y2 = self.movec_plan_middle_cart_position[1]
        z2 = self.movec_plan_middle_cart_position[2]

        x3 = self.movec_plan_target_cart_position[0]
        y3 = self.movec_plan_target_cart_position[1]
        z3 = self.movec_plan_target_cart_position[2]

        ff1 = x1 * x1 + y1 * y1 + z1 * z1
        ff2 = x2 * x2 + y2 * y2 + z2 * z2
        ff3 = x3 * x3 + y3 * y3 + z3 * z3
        ee1 = x3 * z2 - x3 * z1
        ee2 = -x1 * z2 - x2 * z3
        ee3 = x2 * z1 + x1 * z3

        D = z2 * y1 - y2 * z1 + y2 * z3 - z2 * y3 - y1 * z3 + z1 * y3
        E = ee1 + ee2 + ee3
        F = x3 * y1 - x3 * y2 + x1 * y2 - x2 * y1 - x1 * y3 + x2 * y3
        G = x3 * y2 * z1 - x3 * z2 * y1 - x2 * z1 * y3 + x2 * y1 * z3 + x1 * z2 * y3 - x1 * y2 * z3
        H = 2 * (x2 - x1)
        I = 2 * (y2 - y1)
        J = 2 * (z2 - z1)
        K = ff1 - ff2
        O = 2 * (x3 - x1)
        P = 2 * (y3 - y1)
        Q = 2 * (z3 - z1)
        R = ff1 - ff3
        MM = -D * Q * I + Q * H * E + O * F * I - O * E * J - P * H * F + D * P * J

        if abs(MM) > self.MIN_VAL:
            self.movec_plan_center[0] = -(-E * J * R + E * K * Q - F * P * K + F * R * I - G * Q * I + G * P * J) / MM
            self.movec_plan_center[1] = -(D * J * R - D * K * Q - H * F * R - O * J * G + O * K * F + H * G * Q) / MM
            self.movec_plan_center[2] = -(O * G * I + D * P * K - D * R * I + R * H * E - O * E * K - P * H * G) / MM
        else:
            raise ValueError("无法计算圆心位置，请确认设置的三个点是否有点位重合")

        # 计算左臂圆弧运动中圆弧对应的半径长度
        if(np.array_equal(self.movec_plan_current_cart_position, self.movec_plan_center) or np.array_equal(self.movec_plan_middle_cart_position, self.movec_plan_center) or np.array_equal(self.movec_plan_target_cart_position, self.movec_plan_center)):
            raise ValueError("请确认圆心位置是否与设置的三个点位相同，理论上是不同的！！！！！")
        self.movec_plan_radius = math.sqrt((self.movec_plan_current_cart_position[0] - self.movec_plan_center[0]) ** 2 + (self.movec_plan_current_cart_position[1] - self.movec_plan_center[1]) ** 2 + (self.movec_plan_current_cart_position[2] - self.movec_plan_center[2]) ** 2)

        # 计算左臂圆弧运动中圆弧对应的圆心角
        vector_start = (self.movec_plan_current_cart_position[0] - self.movec_plan_center[0], self.movec_plan_current_cart_position[1] - self.movec_plan_center[1], self.movec_plan_current_cart_position[2] - self.movec_plan_center[2])
        vector_end = (self.movec_plan_target_cart_position[0] - self.movec_plan_center[0], self.movec_plan_target_cart_position[1] - self.movec_plan_center[1], self.movec_plan_target_cart_position[2] - self.movec_plan_center[2])
        dot_product = vector_start[0] * vector_end[0] + vector_start[1] * vector_end[1] + vector_start[2] * vector_end[2]
        magnitude_start = math.sqrt(vector_start[0] ** 2 + vector_start[1] ** 2 + vector_start[2] ** 2)
        magnitude_end = math.sqrt(vector_end[0] ** 2 + vector_end[1] ** 2 + vector_end[2] ** 2)
        cos_angle = dot_product / (magnitude_start * magnitude_end)
        self.movec_plan_theta = math.acos(cos_angle)

        # 计算左臂圆弧运动中圆弧对应的圆弧长度 用于规划
        self.movec_plan_arc_length = self.movec_plan_radius * self.movec_plan_theta
        self.movec_plan_displacement = self.movec_plan_arc_length

        # 计算左臂圆弧运动中圆弧对应的旋转轴长度
        v1 = np.array(self.movec_plan_current_cart_position)- np.array(self.movec_plan_middle_cart_position)
        v2 = np.array(self.movec_plan_target_cart_position) - np.array(self.movec_plan_middle_cart_position)
        normal_vector = np.cross(v2, v1)
        normal_vector = np.array(normal_vector)
        self.movec_plan_center_vec = normal_vector / np.linalg.norm(normal_vector)

        # 不考虑中间点的姿态
        if np.dot(self.movec_plan_current_cart_quat.components, self.movec_plan_target_cart_quat.components) < 0:
            self.movec_plan_target_cart_quat = - self.movec_plan_target_cart_quat        


    def cal_right_arm_movec_plan_data(self):
        # 获取右臂当前位置对应的末端笛卡尔位置姿态和四元数
        self.right_arm_movec_plan_current_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movec_plan_current_joint_position[7:14]))
        self.right_arm_movec_plan_current_cart_position = self.right_arm_movec_plan_current_cart.translation
        self.right_arm_movec_plan_current_cart_pose = self.right_arm_movec_plan_current_cart.rotation
        self.right_arm_movec_plan_current_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movec_plan_current_cart_pose)
        if self.whether_save_movec_position:
            print("self.right_arm_movec_plan_current_cart_position  = {} ".format(self.right_arm_movec_plan_current_cart_position))
            print("self.right_arm_movec_plan_current_cart_pose  = {} ".format(self.right_arm_movec_plan_current_cart_pose))

        # 获取右臂中间点位置对应的末端笛卡尔位置姿态和四元数
        self.right_arm_movec_plan_middle_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movec_plan_middle_joint_position[7:14]))
        self.right_arm_movec_plan_middle_cart_position = self.right_arm_movec_plan_middle_cart.translation
        self.right_arm_movec_plan_middle_cart_pose = self.right_arm_movec_plan_middle_cart.rotation
        self.right_arm_movec_plan_middle_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movec_plan_middle_cart_pose)
        if self.whether_save_movec_position:
            print("self.right_arm_movec_plan_middle_cart_position  = {} ".format(self.right_arm_movec_plan_middle_cart_position))
            print("self.right_arm_movec_plan_middle_cart_pose  = {} ".format(self.right_arm_movec_plan_middle_cart_pose))        

        # 获取右臂期望位置对应的末端笛卡尔位置姿态和四元数
        self.right_arm_movec_plan_target_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movec_plan_target_joint_position[7:14]))
        self.right_arm_movec_plan_target_cart_position = self.right_arm_movec_plan_target_cart.translation
        self.right_arm_movec_plan_target_cart_pose = self.right_arm_movec_plan_target_cart.rotation
        self.right_arm_movec_plan_target_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movec_plan_target_cart_pose)
        if self.whether_save_movec_position:
            print("self.right_arm_movec_plan_target_cart_position  = {} ".format(self.right_arm_movec_plan_target_cart_position))
            print("self.right_arm_movec_plan_target_cart_pose  = {} ".format(self.right_arm_movec_plan_target_cart_pose))        

        # 三点位置校验  如何实现单臂中一个运行 一个不运动 这个地方的校验会直接退出 无法实现一个动 一个不动
        if(np.array_equal(self.right_arm_movec_plan_current_cart_position, self.right_arm_movec_plan_middle_cart_position) or np.array_equal(self.right_arm_movec_plan_middle_cart_position, self.right_arm_movec_plan_target_cart_position) or np.array_equal(self.right_arm_movec_plan_current_cart_position, self.right_arm_movec_plan_target_cart_position)):
            raise ValueError("右臂设置的三个点位不能存在两个点完全重合，请重新设置圆弧的中间点位和目标点位")
        

        # 计算圆弧运动中圆弧对应的圆心位置
        x1 = self.right_arm_movec_plan_current_cart_position[0]
        y1 = self.right_arm_movec_plan_current_cart_position[1]
        z1 = self.right_arm_movec_plan_current_cart_position[2]

        x2 = self.right_arm_movec_plan_middle_cart_position[0]
        y2 = self.right_arm_movec_plan_middle_cart_position[1]
        z2 = self.right_arm_movec_plan_middle_cart_position[2]

        x3 = self.right_arm_movec_plan_target_cart_position[0]
        y3 = self.right_arm_movec_plan_target_cart_position[1]
        z3 = self.right_arm_movec_plan_target_cart_position[2]

        ff1 = x1 * x1 + y1 * y1 + z1 * z1
        ff2 = x2 * x2 + y2 * y2 + z2 * z2
        ff3 = x3 * x3 + y3 * y3 + z3 * z3
        ee1 = x3 * z2 - x3 * z1
        ee2 = -x1 * z2 - x2 * z3
        ee3 = x2 * z1 + x1 * z3

        D = z2 * y1 - y2 * z1 + y2 * z3 - z2 * y3 - y1 * z3 + z1 * y3
        E = ee1 + ee2 + ee3
        F = x3 * y1 - x3 * y2 + x1 * y2 - x2 * y1 - x1 * y3 + x2 * y3
        G = x3 * y2 * z1 - x3 * z2 * y1 - x2 * z1 * y3 + x2 * y1 * z3 + x1 * z2 * y3 - x1 * y2 * z3
        H = 2 * (x2 - x1)
        I = 2 * (y2 - y1)
        J = 2 * (z2 - z1)
        K = ff1 - ff2
        O = 2 * (x3 - x1)
        P = 2 * (y3 - y1)
        Q = 2 * (z3 - z1)
        R = ff1 - ff3
        MM = -D * Q * I + Q * H * E + O * F * I - O * E * J - P * H * F + D * P * J

        if abs(MM) > self.MIN_VAL:
            self.right_arm_movec_plan_center[0] = -(-E * J * R + E * K * Q - F * P * K + F * R * I - G * Q * I + G * P * J) / MM
            self.right_arm_movec_plan_center[1] = -(D * J * R - D * K * Q - H * F * R - O * J * G + O * K * F + H * G * Q) / MM
            self.right_arm_movec_plan_center[2] = -(O * G * I + D * P * K - D * R * I + R * H * E - O * E * K - P * H * G) / MM
        else:
            raise ValueError("无法计算圆心位置，请确认设置的三个点是否有点位重合")

        # 计算右臂圆弧运动中圆弧对应的半径长度
        if(np.array_equal(self.right_arm_movec_plan_current_cart_position, self.right_arm_movec_plan_center) or np.array_equal(self.right_arm_movec_plan_middle_cart_position, self.right_arm_movec_plan_center) or np.array_equal(self.right_arm_movec_plan_target_cart_position, self.right_arm_movec_plan_center)):
            raise ValueError("请确认圆心位置是否与设置的三个点位相同，理论上是不同的！！！！！")
        self.right_arm_movec_plan_radius = math.sqrt((self.right_arm_movec_plan_current_cart_position[0] - self.right_arm_movec_plan_center[0]) ** 2 + (self.right_arm_movec_plan_current_cart_position[1] - self.right_arm_movec_plan_center[1]) ** 2 + (self.right_arm_movec_plan_current_cart_position[2] - self.right_arm_movec_plan_center[2]) ** 2)
        # 计算右臂圆弧运动中圆弧对应的圆心角
        vector_start = (self.right_arm_movec_plan_current_cart_position[0] - self.right_arm_movec_plan_center[0], self.right_arm_movec_plan_current_cart_position[1] - self.right_arm_movec_plan_center[1], self.right_arm_movec_plan_current_cart_position[2] - self.right_arm_movec_plan_center[2])
        vector_end = (self.right_arm_movec_plan_target_cart_position[0] - self.right_arm_movec_plan_center[0], self.right_arm_movec_plan_target_cart_position[1] - self.right_arm_movec_plan_center[1], self.right_arm_movec_plan_target_cart_position[2] - self.right_arm_movec_plan_center[2])
        dot_product = vector_start[0] * vector_end[0] + vector_start[1] * vector_end[1] + vector_start[2] * vector_end[2]
        magnitude_start = math.sqrt(vector_start[0] ** 2 + vector_start[1] ** 2 + vector_start[2] ** 2)
        magnitude_end = math.sqrt(vector_end[0] ** 2 + vector_end[1] ** 2 + vector_end[2] ** 2)
        cos_angle = dot_product / (magnitude_start * magnitude_end)
        self.right_arm_movec_plan_theta = math.acos(cos_angle)

        # 计算右臂圆弧运动中圆弧对应的圆弧长度 用于规划
        self.right_arm_movec_plan_arc_length = self.right_arm_movec_plan_radius * self.right_arm_movec_plan_theta
        self.right_arm_movec_plan_displacement = self.right_arm_movec_plan_arc_length

        # 计算右臂圆弧运动中圆弧对应的旋转轴长度
        v1 = np.array(self.right_arm_movec_plan_current_cart_position)- np.array(self.right_arm_movec_plan_middle_cart_position)
        v2 = np.array(self.right_arm_movec_plan_target_cart_position) - np.array(self.right_arm_movec_plan_middle_cart_position)
        normal_vector = np.cross(v2, v1)
        normal_vector = np.array(normal_vector)
        self.right_arm_movec_plan_center_vec = normal_vector / np.linalg.norm(normal_vector)

        # 不考虑中间点的姿态
        if np.dot(self.right_arm_movec_plan_current_cart_quat.components, self.right_arm_movec_plan_target_cart_quat.components) < 0:
            self.right_arm_movec_plan_target_cart_quat = - self.right_arm_movec_plan_target_cart_quat        



    def hybrid_force_movec_plan_interpolation(self):
        """
        执行力位混合控制的参数计算与圆弧速度规划后，执行此函数进行插值和数据的实时发送
        """
        self.collision_detection.start_collision_detection()


        for interpolation_time in np.arange(0, self.speed_plan.time_length, self.interpolation_period / 1000):
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
            self.movec_plan_interpolation_theta_inter = self.speed_plan.cur_disp_normalization_ratio * self.movec_plan_theta
            self.cart_interpolation_position = self.left_arm_circle_traj_interpolation()
            # 将插补位置映射到力的法平面上(当目标力不为0的时候)
            if np.linalg.norm(self.left_arm_target_FT_data) > self.MIN_VAL:
                self.project_left_arm_traj_interpolation()


            
            # 恒力控制部分分量
            # 读取当前末端力数据
            current_FT = self.force_control_data.left_arm_FT_original_MAF_compensation_base_coordinate_system[:3]
            target_FT = self.left_arm_target_FT_data[:3]  # 只提取轴向目标力分量
            err_F_num = np.dot(target_FT - current_FT, target_FT) / np.linalg.norm(target_FT)

            # 采用导纳控制计算末端位移（将三分力看作一维，避免对轨迹规划部分产生影响）
            self.left_arm_effector_current_acc = (err_F_num - self.left_arm_admittance_control_B * np.linalg.norm(
                self.left_arm_effector_pre_acc)) / self.left_arm_admittance_control_M
            self.left_arm_effector_current_acc = 0.5 * (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc)

            self.left_arm_effector_current_speed = (self.left_arm_effector_current_acc + self.left_arm_effector_pre_acc) * (
                        self.interpolation_period / 1000)
            self.left_arm_effector_current_speed = 0.5 * self.left_arm_effector_current_speed + 0.5 * self.left_arm_effector_pre_speed

            # 将导纳部分的位移增量映射到力的方向
            if np.linalg.norm(self.left_arm_target_FT_data) > self.MIN_VAL:
                cart_position_increment_2 = self.left_arm_effector_current_speed * (self.interpolation_period / 1000) * target_FT / np.linalg.norm(target_FT)
            else:
                cart_position_increment_2 = np.zeros(3)
            self.cart_interpolation_position += cart_position_increment_2

            # 四元数球面线性插值
            slerped_quaternions = quaternion.slerp(self.movec_plan_current_cart_quat, self.movec_plan_target_cart_quat, 0, 1, self.speed_plan.cur_disp_normalization_ratio)
            self.cart_interpolation_pose = quaternion.as_rotation_matrix(slerped_quaternions)

            #  左臂逆解 逆解 逆解
            self.kinematic_model.left_arm_inverse_kinematics(self.cart_interpolation_pose, self.cart_interpolation_position,
                                                             self.movec_plan_current_joint_position[:7])

            # 计算右臂插补位置与姿态
            # 轨迹规划部分分量
            self.right_arm_movec_plan_interpolation_theta_inter = self.speed_plan.cur_disp_normalization_ratio * self.right_arm_movec_plan_theta
            self.right_arm_cart_interpolation_position = self.right_arm_circle_traj_interpolation()
            # 将插补位置映射到力的法平面上
            if np.linalg.norm(self.right_arm_target_FT_data) > self.MIN_VAL:
                self.project_right_arm_traj_interpolation()


            # 读取当前末端力数据
            current_FT = self.force_control_data.right_arm_FT_original_MAF_compensation_base_coordinate_system[:3]
            target_FT = self.right_arm_target_FT_data[:3]  # 只提取轴向目标力分量
            err_F_num = np.dot(target_FT - current_FT, target_FT) / np.linalg.norm(target_FT)

            # 采用导纳控制计算末端位移（将三分力看作一维，避免对轨迹规划部分产生影响）
            self.right_arm_effector_current_acc = (err_F_num - self.right_arm_admittance_control_B * np.linalg.norm(
                self.right_arm_effector_pre_acc)) / self.right_arm_admittance_control_M
            self.right_arm_effector_current_acc = 0.5 * (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc)

            self.right_arm_effector_current_speed = (self.right_arm_effector_current_acc + self.right_arm_effector_pre_acc) * (
                        self.interpolation_period / 1000)
            self.right_arm_effector_current_speed = 0.5 * self.right_arm_effector_current_speed + 0.5 * self.right_arm_effector_pre_speed

            # 将导纳部分的位移增量映射到力的方向
            if np.linalg.norm(self.right_arm_target_FT_data) > self.MIN_VAL:
                cart_position_increment_2 = self.right_arm_effector_current_speed * (self.interpolation_period / 1000) * target_FT / np.linalg.norm(target_FT)
            else:
                cart_position_increment_2 = np.zeros(3)
            self.right_arm_cart_interpolation_position += cart_position_increment_2

            slerped_quaternions = quaternion.slerp(self.right_arm_movec_plan_current_cart_quat,
                                                   self.right_arm_movec_plan_target_cart_quat, 0, 1, self.speed_plan.cur_disp_normalization_ratio)
            self.right_arm_cart_interpolation_pose = quaternion.as_rotation_matrix(slerped_quaternions)

            # # 右臂逆解 逆解 逆解
            self.kinematic_model.right_arm_inverse_kinematics(self.right_arm_cart_interpolation_pose,
                                                              self.right_arm_cart_interpolation_position,
                                                              self.movec_plan_current_joint_position[7:14])

            # 校验逆解是否成功
            if (self.kinematic_model.left_arm_inverse_kinematics_solution_success_flag and self.kinematic_model.right_arm_inverse_kinematics_solution_success_flag) == False:
                print("逆解失败咯，机器人应该停止运行了，请调整合理的位置呀！！！！")
                self.interpolation_result = self.movec_plan_current_joint_position
                break
            else:
                self.interpolation_result = self.movec_plan_current_joint_position
                self.interpolation_result[7:14] = self.kinematic_model.right_arm_interpolation_result
                self.interpolation_result[:7] = self.kinematic_model.left_arm_interpolation_result

            # 保存运动数据
            if self.whether_save_movec_position:
                self.interpolation_result_cart = deepcopy(
                    self.kinematic_model.left_arm_forward_kinematics(self.interpolation_result[:7]))
                self.interpolation_result_cart_position = self.interpolation_result_cart.translation
                self.interpolation_result_cart_pose = self.interpolation_result_cart.rotation
                self.interpolation_result_cart_quat = quaternion.from_rotation_matrix(self.interpolation_result_cart_pose)

                with open("movec_interpolate_trajectory.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result)

                with open("movec_left_arm_interpolation_result_cart_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result_cart_position)

                with open("movec_left_arm_interpolation_result_cart_pose.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result_cart_pose)

                self.right_arm_interpolation_result_cart = deepcopy(
                    self.kinematic_model.right_arm_forward_kinematics(self.interpolation_result[7:14]))
                self.right_arm_interpolation_result_cart_position = self.right_arm_interpolation_result_cart.translation
                self.right_arm_interpolation_result_cart_pose = self.right_arm_interpolation_result_cart.rotation
                self.right_arm_interpolation_result_cart_quat = quaternion.from_rotation_matrix(
                    self.right_arm_interpolation_result_cart_pose)

                with open("movec_right_arm_interpolation_result_cart_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.right_arm_interpolation_result_cart_position)

                with open("movec_right_arm_interpolation_result_cart_pose.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.right_arm_interpolation_result_cart_pose)

            # 碰撞检测
            if (self.collision_detection.collision_detection_index):
                print("发生了碰撞，结束碰撞检测线程，退出当前插补函数！！！！")
                self.collision_detection.stop_collision_detection()
                sys.exit()  # 退出程序循环，机械臂停止运动

            # 数据发布
            # 双臂关节数据
            self.lcm_hancle.upper_body_data_publisher(self.interpolation_result)
            self.movec_plan_current_joint_position = self.interpolation_result

            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间

        print("运行结束，到达目标点位！！！")
        self.collision_detection.stop_collision_detection()

    def left_arm_circle_traj_interpolation(self):
        a = np.cos(self.movec_plan_interpolation_theta_inter / 2)
        b, c, d = - self.movec_plan_center_vec * np.sin(self.movec_plan_interpolation_theta_inter / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d        
        rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        
        interpolation_point = np.dot(rotation_matrix, np.array(self.movec_plan_current_cart_position) - np.array(self.movec_plan_center)) + np.array(self.movec_plan_center)
        return interpolation_point
    

    def right_arm_circle_traj_interpolation(self):
        a = np.cos(self.right_arm_movec_plan_interpolation_theta_inter / 2)
        b, c, d = - self.right_arm_movec_plan_center_vec * np.sin(self.right_arm_movec_plan_interpolation_theta_inter / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d        
        rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        
        interpolation_point = np.dot(rotation_matrix, np.array(self.right_arm_movec_plan_current_cart_position) - np.array(self.right_arm_movec_plan_center)) + np.array(self.right_arm_movec_plan_center)
        return interpolation_point



    def project_left_arm_traj_interpolation(self):
        normal = self.cart_interpolation_position / np.linalg.norm(self.cart_interpolation_position)
        distance = np.dot(self.left_arm_target_FT_data, normal)
        self.cart_interpolation_position = self.cart_interpolation_position - distance * normal


    def project_right_arm_traj_interpolation(self):
        normal = self.right_arm_cart_interpolation_position / np.linalg.norm(self.right_arm_cart_interpolation_position)
        distance = np.dot(self.right_arm_target_FT_data, normal)
        self.right_arm_cart_interpolation_position = self.right_arm_cart_interpolation_position - distance * normal


    def hybrid_force_movec_control_by_joint(self, current_position, middle_position, target_position, target_FT_data):
        """
        双臂混合力控制+位置控制,以关节角作为输入
        """
        self.cal_hybrid_force_movec_plan_data_by_joint(current_position, middle_position, target_position, target_FT_data)
        self.speed_plan = seven_segment_speed_plan(self.movec_plan_jerk_max, self.movec_plan_acc_max, self.movec_plan_speed_max, max(self.right_arm_movec_plan_displacement, self.movec_plan_displacement))
        self.hybrid_force_movec_plan_interpolation()


    def hybrid_force_movec_control_by_cart(self, left_current_position, left_middle_position, left_target_position, 
                                   right_current_position, right_middle_position, right_target_position, 
                                   target_FT_data, threshold):
        """
        双臂混合力控制+位置控制,以笛卡尔位姿作为输入
        """

        self.cal_hybrid_force_movec_plan_data_by_cart(left_current_position, left_middle_position, left_target_position, 
                                                      right_current_position, right_middle_position, right_target_position,
                                                      target_FT_data)
        self.speed_plan = seven_segment_speed_plan(self.movec_plan_jerk_max, self.movec_plan_acc_max, self.movec_plan_speed_max, max(self.right_arm_movec_plan_displacement, self.movec_plan_displacement))
        self.hybrid_force_movec_plan_interpolation()