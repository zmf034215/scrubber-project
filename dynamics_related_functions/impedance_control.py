import threading
import numpy as np
import time
from os.path import dirname, join, abspath
import os
import pinocchio

from copy import deepcopy

class Impedance_Control():
    def __init__(self):
        self.impedance_control_flag = 0    # 设置阻抗控制启用与否的标志

        self.joint_current_acc =  np.zeros(30)
        
        self.alpha_actual = 0.05    # 实际位置、速度、加速度滤波系数
        self.alpah_impedance = 0.5

        # 阻抗参数（6维：3位置 + 3姿态）
        self.Md = np.diag([0.05]*6)
        self.Dd = np.diag([1]*6)  # 小于10
        self.Kd = np.diag([1]*6)  # 小于10

        self.pre_actual_torq =  np.zeros(30)
        self.pre_actual_posi =  np.zeros(30)
        self.pre_actual_speed =  np.zeros(30)
        self.pre_actual_acc =  np.zeros(30)

        self.lcm_from_robot_period = 2  # LCM 数据发送周期 (ms)

        parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(parent_folder, "models", "z2_left_arm.urdf")
        self.left_arm_pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.left_arm_pin_data = self.left_arm_pin_model.createData()
        
        urdf_path = os.path.join(parent_folder, "models", "z2_right_arm.urdf")
        self.right_arm_pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.right_arm_pin_data = self.right_arm_pin_model.createData()
        
    def impedance_controller(self, x, dx, x_d, dx_d, ddx_d):  #后期可进行扩展：从末端力传感器获取外力F_ext，并合并至F_total中
        x_err = x - x_d
        dx_err = dx - dx_d                            
        F_des = self.Md @ ddx_d + self.Dd @ dx_err + self.Kd @ x_err
        F_total = F_des 
        return F_total  # 6*1
                            
    def get_end_effector_state_left(self, q, dq , ddq):   # 雅可比+正运动学：获取末端执行器的位置+姿态——左臂
        q = np.array(q)
        dq = np.array(dq)
        ddq = np.array(ddq)
        pinocchio.forwardKinematics(self.left_arm_pin_model, self.left_arm_pin_data, q, dq)
        pinocchio.updateFramePlacements(self.left_arm_pin_model, self.left_arm_pin_data)
        frame_id_left = self.left_arm_pin_model.getFrameId("link_la5")
        placement_left = self.left_arm_pin_data.oMf[frame_id_left]
        x_left_posi = placement_left.translation
        x_left_rpy = pinocchio.utils.matrixToRpy(placement_left.rotation)
        x_left = np.array(x_left_posi.tolist() + x_left_rpy.tolist())
        # 笛卡尔速度 = J * dq
        J_left = pinocchio.computeFrameJacobian(self.left_arm_pin_model, self.left_arm_pin_data, q, frame_id_left)
        pinocchio.computeJointJacobiansTimeVariation(self.left_arm_pin_model, self.left_arm_pin_data, q, dq)
        dJ_left = pinocchio.getFrameJacobianTimeVariation(self.left_arm_pin_model, self.left_arm_pin_data, frame_id_left, pinocchio.ReferenceFrame.LOCAL)
        ddx_left = J_left @ ddq + dJ_left @  dq
        dx_left = J_left @ dq
        return x_left, dx_left, J_left , ddx_left

    def get_end_effector_state_right(self, q, dq , ddq):   # 雅可比+正运动学：获取末端执行器的位置+姿态——右臂
        q = np.array(q)
        dq = np.array(dq)
        ddq = np.array(ddq)
        pinocchio.forwardKinematics(self.right_arm_pin_model, self.right_arm_pin_data, q, dq)
        pinocchio.updateFramePlacements(self.right_arm_pin_model, self.right_arm_pin_data)
        frame_id_right = self.right_arm_pin_model.getFrameId("link_ra5")
        placement_right = self.right_arm_pin_data.oMf[frame_id_right]
        x_right_posi = placement_right.translation
        x_right_rpy = pinocchio.utils.matrixToRpy(placement_right.rotation)
        x_right = np.array(x_right_posi.tolist() + x_right_rpy.tolist())
        # 笛卡尔速度 = J * dq
        J_right = pinocchio.computeFrameJacobian(self.right_arm_pin_model, self.right_arm_pin_data, q, frame_id_right)
        pinocchio.computeJointJacobiansTimeVariation(self.right_arm_pin_model, self.right_arm_pin_data, q, dq)
        dJ_right = pinocchio.getFrameJacobianTimeVariation(self.right_arm_pin_model, self.right_arm_pin_data, frame_id_right, pinocchio.ReferenceFrame.LOCAL)
        ddx_right = J_right @ ddq + dJ_right @  dq
        dx_right = J_right @ dq
        return x_right, dx_right, J_right, ddx_right

    def impedance_jointTorq(self, actual_position_0,actual_speed_0, position_desired, speed_desired, acc_desired) :  #(实际位置，实际速度，期望位置，期望速度，期望加速度 14*1)
        if np.sum(self.pre_actual_speed) == 0 :
            actual_acc_0 = np.zeros(30).tolist() 
        else :
            actual_acc_0 = (actual_speed_0 - self.pre_actual_speed) / (self.lcm_from_robot_period / 1000)
        actual_position = np.zeros(30).tolist()   # 30*1
        actual_speed = np.zeros(30).tolist()
        actual_acc = np.zeros(30).tolist()
        # 对实际数据滤波
        actual_position[:14] = np.array(self.pre_actual_posi[:14]) * (1 - self.alpha_actual) + ( self.alpha_actual * actual_position_0[:14])
        actual_speed[:14] = np.array(self.pre_actual_speed[:14]) * (1 - self.alpha_actual) + ( self.alpha_actual * actual_speed_0[:14])
        actual_acc[:14] = np.array(self.pre_actual_acc[:14]) * (1 - self.alpha_actual) + ( self.alpha_actual * np.array(actual_acc_0[:14]))
        
        x_left_actual,dx_left_actual,J_left_actual,ddx_left_actual = self.get_end_effector_state_left(actual_position[:5],actual_speed[:5],actual_acc[:5])
        x_left_desired,dx_left_desired,J_left_desired,ddx_left_desired = self.get_end_effector_state_left(position_desired[:5],speed_desired[:5],acc_desired[:5])
        F_total_left = self.impedance_controller(x_left_actual,dx_left_actual,x_left_desired,dx_left_desired,ddx_left_desired)
        left_JointTorqueVec_impe = J_left_actual.T @ F_total_left   # 5*1

        x_right_actual,dx_right_actual,J_right_actual,ddx_right_actual = self.get_end_effector_state_right(actual_position[7:12],actual_speed[7:12],actual_acc[7:12])
        x_right_desired,dx_right_desired,J_right_desired,ddx_right_desired = self.get_end_effector_state_right(position_desired[7:12],speed_desired[7:12],acc_desired[7:12])
        F_total_right = self.impedance_controller(x_right_actual,dx_right_actual,x_right_desired,dx_right_desired,ddx_right_desired)
        right_JointTorqueVec_impe = J_right_actual.T @ F_total_right   # 5*1

        zeros0 = np.array([0, 0])
        JointTorqueVec_impe = np.hstack((left_JointTorqueVec_impe, zeros0, right_JointTorqueVec_impe, zeros0))   #14*1
        
        # jointTorqueVec_impe_all = np.array(actual_jointTorqueVec[:14]) + JointTorqueVec_impe   #  动力学前馈补偿 + 阻抗力补偿
        # jointTorqueVec_impe_all = np.concatenate((jointTorqueVec,actual_jointTorqueVec[14:]))
        jointTorqueVec_impe_all = np.copy(JointTorqueVec_impe)
        jointTorqueVec_impe_all = self.pre_actual_torq[:14] * (1 - self.alpah_impedance) + self.alpah_impedance * jointTorqueVec_impe_all    # 阻抗转矩滤波
        
        self.pre_actual_posi = np.copy(actual_position)
        self.pre_actual_speed = np.copy(actual_speed) 
        # self.pre_actual_speed_0 = np.copy(actual_speed_0)
        self.pre_actual_acc = np.copy(actual_acc) 
        self.pre_actual_torq[:14] = np.copy(jointTorqueVec_impe_all)
        
        # self.pre_target_torq = np.copy(target_jointTorqueVec)
        
        return jointTorqueVec_impe_all     #14*1




