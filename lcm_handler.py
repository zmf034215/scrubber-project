from lcm_data_structure.upper_body_cmd_package import upper_body_cmd_package
from lcm_data_structure.upper_body_data_package import upper_body_data_package
from lcm_data_structure.ecat_debug_ft_data_lcmt import ecat_debug_ft_data_lcmt

import lcm
import threading
import numpy as np
import time

import pinocchio
import os

from copy import deepcopy

class LCMHandler:
    def __init__(self):
        """
        初始化 LCM 处理类
        """

        current_folder = os.getcwd()
        urdf_path = os.path.join(current_folder, "models/p5_humanoid_v1.0.urdf")

        # 加载机器人模型和数据
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()


        # 关节空间下的位置和速度以及upper_body_data包中的数据
        self.dim = 30
        self.joint_current_pos =  np.zeros(30)

        self.plan_pre_qpos = None
        self.plan_pre_speed = None
        self.plan_pre_acc = None

        self.is_used = None
        self.error_code = None
        self.status = None
        self.joint_current_speed = None
        self.joint_current_current_or_torque = None
        self.data_lock = threading.Lock() 
        self.left_arm_joint_pre_speed = np.array([0, 0, 0, 0, 0, 0, 0])
        self.right_arm_joint_pre_speed = np.array([0, 0, 0, 0, 0, 0, 0])

        # 力传感器原始数据
        self.left_arm_FT_original =[0.0 for i in range(6)]
        self.right_arm_FT_original =[0.0 for i in range(6)]

        self.interpolation_period = 2   # 单位是ms


        # 线程同步机制
        self.joint_current_pos_updated = threading.Event()  # 标记数据已更新
        self.joint_current_pos_lock = threading.Lock()  # 线程锁，防止数据竞争

        # LCM 通信参数
        self.lcm_from_robot_period = 2  # LCM 数据发送周期 (ms)
        self.lcm_form_robot_fs = 1000 / self.lcm_from_robot_period
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=1')

        # 订阅 LCM 话题  如果对应的话题没有被发布 则对应的回调函数就不会被执行
        self.lcm.subscribe('upper_body_data', self.upper_body_data_listener)
        self.lcm.subscribe('ecat_debug_FT_dataARM_FT_L', self.FT_left_data_listener)
        self.lcm.subscribe('ecat_debug_FT_dataARM_FT_R', self.FT_right_data_listener)

        # 上半身伺服模式设置 用于upper_body_cmd数据下发
        self.default_arm_control_mode = [ 4 for dim0 in range(14) ]
        self.default_hand_control_mode = [ 4 for dim0 in range(12) ]
        self.default_waist_control_mode = [ 4 for dim0 in range(2) ]
        self.default_head_control_mode = [ 4 for dim0 in range(2) ]
        
        # 两个手臂的六七关节伺服模式设置为模式5
        self.default_arm_control_mode[5] = 5
        self.default_arm_control_mode[6] = 5
        self.default_arm_control_mode[12] = 5
        self.default_arm_control_mode[13] = 5
        self.default_control_mode = self.default_arm_control_mode \
                                    + self.default_hand_control_mode \
                                    + self.default_waist_control_mode \
                                    + self.default_head_control_mode




        # 力传感器数据滤波缓存区
        self.left_arm_FT_original_buff_size = 20
        self.left_arm_FT_original_buff = [[0] * 6 for _ in range(self.left_arm_FT_original_buff_size)]
        self.right_arm_FT_original_buff_size = 20
        self.right_arm_FT_original_buff = [[0] * 6 for _ in range(self.right_arm_FT_original_buff_size)]

        # 力传感器数据滤波后的结果
        self.left_arm_FT_original_MAF = [0.0 for i in range(6)]
        self.right_arm_FT_original_MAF = [0.0 for i in range(6)]

        # 力传感器数据滤波补偿后的结果
        self.left_arm_FT_original_MAF_compensation = [0.0 for i in range(6)]
        self.right_arm_FT_original_MAF_compensation = [0.0 for i in range(6)]

        self.left_arm_FT_original_MAF_compensation_2 = [0.0 for i in range(6)]
        self.right_arm_FT_original_MAF_compensation_2 = [0.0 for i in range(6)]

        # 力传感器数据滤波补偿后转换到基坐标系下的结果
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system = np.array([0.0 for i in range(6)])
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system = np.array([0.0 for i in range(6)])

        self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre = np.array([0.0 for i in range(6)])
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system_pre = np.array([0.0 for i in range(6)])


        self.left_arm_FT_original_MAF_compensation_base_coordinate_system_2 = np.array([0.0 for i in range(6)])

        self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre_2 = np.array([0.0 for i in range(6)])

        ## 传感器标定结果 在更换工装 或者传感器数据不准时 执行FT_data_calibration 把打印的结果替换掉下面的变量
        self.left_arm_force_sensor_U =  -0.022038554065862543
        self.left_arm_force_sensor_V =  -0.035732404863668174
        self.left_arm_force_sensor_G =  4.7192077497213445
        self.left_arm_force_sensor_data_L =  [-0.16855181901619956, 0.10399609621622885, -4.715050040170105]
        self.left_arm_force_sensor_com =  [0.013996203651423768, -0.029254033370361296, -0.035436677023890806]
        self.left_arm_force_sensor_data_Foffset =  [2.422074974285764, 1.499556791513176, 3.157712468951261]
        self.left_arm_force_sensor_data_Moffset =  [-0.03964647477199035, -0.13257402858512296, 0.0999744133336088]
        self.left_arm_force_sensor_mass =  0.4815518111960555
        self.right_arm_force_sensor_U =  0.03643448677285478
        self.right_arm_force_sensor_V =  0.05056155958140244
        self.right_arm_force_sensor_G =  5.005101416226769
        self.right_arm_force_sensor_data_L =  [0.25279004268375993, -0.18231795810623852, -4.99538762692017]
        self.right_arm_force_sensor_com =  [0.030336498158045743, 0.02022538912470985, -0.03984968846092607]
        self.right_arm_force_sensor_data_Foffset =  [2.169360130867161, -2.9570789370786414, 3.3885130740743916]
        self.right_arm_force_sensor_data_Moffset =  [-0.0659167054079941, -0.1656463910222986, -0.13287034834905143]
        self.right_arm_force_sensor_mass =  0.510724634308854

        # 线程启动
        self.lcm_thread_handle = threading.Thread(target=self.lcm_handle, daemon=True)
        self.lcm_thread_handle.start()

    def lcm_handle(self):
        """
        处理 LCM 消息（阻塞式）
        """
        while True:

            self.lcm.handle()


    def upper_body_data_listener(self, channel, data):
        """
        监听机器人关节数据
        """
        msg = upper_body_data_package.decode(data)
        if not hasattr(msg, 'curJointPosVec') or not msg.curJointPosVec:
            print("⚠️ Warning: Received invalid joint position data! Skipping update.")
            return

        with self.joint_current_pos_lock:
            self.joint_current_pos = np.array(msg.curJointPosVec)
            self.is_used = np.array(msg.isUsed)
            self.error_code = np.array(msg.curErrCodeVec)
            self.status = np.array(msg.curStatusVec)
            self.joint_current_speed = np.array(msg.curSpeedVec)
            self.joint_current_current_or_torque = np.array(msg.curCurrentVec)
            self.joint_current_pos_updated.set()


    def FT_left_data_listener(self, channel, data):
        """
        监听左臂力传感器数据
        """
        if not self.joint_current_pos_updated.wait(timeout=0.001):
            print("⚠️ Warning: joint_current_pos not updated, skipping FT data compensation")
            return

        with self.joint_current_pos_lock:
            msg = ecat_debug_ft_data_lcmt.decode(data)

            # 解析六维力传感器数据
            self.left_arm_FT_original = np.array([
                msg.original_Fy, msg.original_Fx, msg.original_Fz,
                msg.original_My, msg.original_Mx, msg.original_Mz
            ])

        self.update_left_arm_FT_original_buff_and_data_filtering()
            

    def FT_right_data_listener(self, channel, data):
        """
        监听右臂力传感器数据
        """
        if not self.joint_current_pos_updated.wait(timeout=0.001):
            print("⚠️ Warning: joint_current_pos not updated, skipping FT data compensation")
            return

        with self.joint_current_pos_lock:
            msg = ecat_debug_ft_data_lcmt.decode(data)

            # 解析六维力传感器数据
            ft_data = np.array([
                msg.original_Fy, msg.original_Fx, msg.original_Fz,
                msg.original_My, msg.original_Mx, msg.original_Mz
            ])

            # 坐标转换（确保方向一致）
            self.right_arm_FT_original = np.array([-ft_data[0], -ft_data[1], ft_data[2],
                                                   -ft_data[3], -ft_data[4], ft_data[5]])

        self.update_right_arm_FT_original_buff_and_data_filtering()



    def upper_body_data_publisher(self, package):
        """
        将输入的关节位姿转换为 LCM 消息
        """
        arm_and_hand_ctrl_msg = upper_body_cmd_package()
        arm_and_hand_ctrl_msg.isUsed = 0
        arm_and_hand_ctrl_msg.control_mode = self.default_control_mode
        arm_and_hand_ctrl_msg.jointPosVec = package.tolist()
        arm_and_hand_ctrl_msg.jointKp = (np.ones(30) * 40).tolist()
        arm_and_hand_ctrl_msg.jointKd = (np.ones(30) * 100).tolist()

        if self.plan_pre_qpos is None:
            speed = np.zeros_like(package)
            acc = np.zeros_like(package)
        else:
            speed = (package - self.plan_pre_qpos) / (self.lcm_from_robot_period / 1000)
            acc = np.zeros_like(package) if np.sum(self.plan_pre_speed) == 0 else \
                (speed - self.plan_pre_speed) / (self.lcm_from_robot_period / 1000)

        arm_and_hand_ctrl_msg.jointSpeedVec = speed.tolist()
        arm_and_hand_ctrl_msg.jointCurrentVec = np.zeros_like(package).tolist()
        arm_and_hand_ctrl_msg.jointTorqueVec = np.zeros_like(package).tolist()

        # 更新历史数据
        self.plan_pre_qpos = np.copy(package)
        self.plan_pre_speed = np.copy(speed)
        self.plan_pre_acc = np.copy(acc)
        self.lcm.publish('upper_body_cmd', arm_and_hand_ctrl_msg.encode())




    def update_left_arm_FT_original_buff_and_data_filtering(self):
        # 插入新的数据
        self.left_arm_FT_original_buff[1:self.left_arm_FT_original_buff_size - 1] = self.left_arm_FT_original_buff[0:self.left_arm_FT_original_buff_size - 2]
        if isinstance(self.left_arm_FT_original, np.ndarray):
            self.left_arm_FT_original_buff[0] = self.left_arm_FT_original.flatten().tolist()
        else:
            self.left_arm_FT_original_buff[0] = self.left_arm_FT_original

        # 均值滤波器
        self.left_arm_FT_original_MAF = self.left_arm_FT_data_moving_average_filter(self.left_arm_FT_original_buff)

        # 传感器数据标定后补偿
        if (self.left_arm_force_sensor_mass != 0):
            self.left_arm_FT_data_compensation()
            self.cal_left_arm_FT_original_MAF_compensation_base_coordinate_system()

    def left_arm_FT_data_moving_average_filter(self, data):

        data = np.array(data)
        num_columns = data.shape[1]
        filtered_data = np.zeros(num_columns)
        
        for i in range(num_columns):
            filtered_data[i] = np.sum(data[:, i]) / self.left_arm_FT_original_buff_size
        
        filtered_data_list = filtered_data.tolist()
        return filtered_data_list

    def left_arm_FT_data_compensation(self):
        if (self.left_arm_force_sensor_G == 0):
            print("请先进行传感器数据的标定！！！")
        else:
            # 基于六维力传感器的工业机器人末端负载受力感知研究 -- 张立建
            qpos_ros = self.qpos_lcm2ros(self.joint_current_pos)
            pinocchio.forwardKinematics(self.model, self.data, qpos_ros)
            # 关节角度更新后更新各个坐标系的位置
            pinocchio.updateFramePlacements(self.model, self.data)    
            # 基坐标系下末端力传感器姿态矩阵的转置矩阵 
            end_frame_ID_left = self.model.getFrameId("link_FT_l")
            R = self.data.oMf[end_frame_ID_left].rotation.T   
            G = R @ self.left_arm_force_sensor_data_L

            self.left_arm_FT_original_MAF_compensation[0] =  self.left_arm_FT_original_MAF[0] - self.left_arm_force_sensor_data_Foffset[0] - G[0]
            self.left_arm_FT_original_MAF_compensation[1] =  self.left_arm_FT_original_MAF[1] - self.left_arm_force_sensor_data_Foffset[1] - G[1]
            self.left_arm_FT_original_MAF_compensation[2] =  self.left_arm_FT_original_MAF[2] - self.left_arm_force_sensor_data_Foffset[2] - G[2]

            self.left_arm_FT_original_MAF_compensation[3] =  self.left_arm_FT_original_MAF[3] - self.left_arm_force_sensor_data_Moffset[0] - (G[2] * self.left_arm_force_sensor_com[1] - G[1] * self.left_arm_force_sensor_com[2])
            self.left_arm_FT_original_MAF_compensation[4] =  self.left_arm_FT_original_MAF[4] - self.left_arm_force_sensor_data_Moffset[1] - (G[0] * self.left_arm_force_sensor_com[2] - G[2] * self.left_arm_force_sensor_com[0])
            self.left_arm_FT_original_MAF_compensation[5] =  self.left_arm_FT_original_MAF[5] - self.left_arm_force_sensor_data_Moffset[2] - (G[1] * self.left_arm_force_sensor_com[0] - G[0] * self.left_arm_force_sensor_com[1])

  



    def cal_left_arm_FT_original_MAF_compensation_base_coordinate_system(self):     # 将末端六维力传感器的数据从TCP坐标系转换到基坐标系下 现代机器人学
        
        qpos_ros = self.qpos_lcm2ros(self.joint_current_pos)

        pinocchio.forwardKinematics(self.model, self.data, qpos_ros)

        end_frame_ID_left = self.model.getFrameId("link_la7")

        target_cart_pose = deepcopy(self.data.oMf[end_frame_ID_left].rotation)

        base_in_effector_pose = target_cart_pose.T
    
    
        base_in_effector_0r = np.hstack(((np.zeros((3, 3))), base_in_effector_pose))
        base_in_effector_r0 = np.hstack((base_in_effector_pose, (np.zeros((3, 3)))))
        base_in_effector_ad = np.vstack((base_in_effector_r0, base_in_effector_0r))
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system = base_in_effector_ad.T @ self.left_arm_FT_original_MAF_compensation
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system = 0.01 * self.left_arm_FT_original_MAF_compensation_base_coordinate_system + 0.99 * self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre = self.left_arm_FT_original_MAF_compensation_base_coordinate_system


    def update_right_arm_FT_original_buff_and_data_filtering(self):
        # 插入新的数据
        self.right_arm_FT_original_buff[1:self.right_arm_FT_original_buff_size - 1] = self.right_arm_FT_original_buff[0:self.right_arm_FT_original_buff_size - 2]
        if isinstance(self.right_arm_FT_original, np.ndarray):
            self.right_arm_FT_original_buff[0] = self.right_arm_FT_original.flatten().tolist()
        else:
            self.right_arm_FT_original_buff[0] = self.right_arm_FT_original

        # 均值滤波器
        self.right_arm_FT_original_MAF = self.right_arm_FT_data_moving_average_filter(self.right_arm_FT_original_buff)
        # 传感器数据标定后补偿
        if (self.right_arm_force_sensor_mass != 0):
            self.right_arm_FT_data_compensation()
            self.cal_right_arm_FT_original_MAF_compensation_base_coordinate_system()

    def right_arm_FT_data_moving_average_filter(self, data):
        data = np.array(data)
        num_columns = data.shape[1]
        filtered_data = np.zeros(num_columns)
        
        for i in range(num_columns):
            filtered_data[i] = np.sum(data[:, i]) / self.right_arm_FT_original_buff_size
        
        filtered_data_list = filtered_data.tolist()
        return filtered_data_list

    def right_arm_FT_data_compensation(self):
        if (self.right_arm_force_sensor_G == 0):
            print("请先进行传感器数据的标定！！！")
        else:
            # 基于六维力传感器的工业机器人末端负载受力感知研究 -- 张立建
            qpos_ros = self.qpos_lcm2ros(self.joint_current_pos)
            pinocchio.forwardKinematics(self.model, self.data, qpos_ros)
            # 关节角度更新后更新各个坐标系的位置
            pinocchio.updateFramePlacements(self.model, self.data)    
            # 基坐标系下末端力传感器姿态矩阵的转置矩阵 

            end_frame_ID_right = self.model.getFrameId("link_FT_r")
            R = self.data.oMf[end_frame_ID_right].rotation.T   
            G = R @ self.right_arm_force_sensor_data_L

            self.right_arm_FT_original_MAF_compensation[0] =  self.right_arm_FT_original_MAF[0] - self.right_arm_force_sensor_data_Foffset[0] - G[0]
            self.right_arm_FT_original_MAF_compensation[1] =  self.right_arm_FT_original_MAF[1] - self.right_arm_force_sensor_data_Foffset[1] - G[1]
            self.right_arm_FT_original_MAF_compensation[2] =  self.right_arm_FT_original_MAF[2] - self.right_arm_force_sensor_data_Foffset[2] - G[2]

            self.right_arm_FT_original_MAF_compensation[3] =  self.right_arm_FT_original_MAF[3] - self.right_arm_force_sensor_data_Moffset[0] - (G[2] * self.right_arm_force_sensor_com[1] - G[1] * self.right_arm_force_sensor_com[2])
            self.right_arm_FT_original_MAF_compensation[4] =  self.right_arm_FT_original_MAF[4] - self.right_arm_force_sensor_data_Moffset[1] - (G[0] * self.right_arm_force_sensor_com[2] - G[2] * self.right_arm_force_sensor_com[0])
            self.right_arm_FT_original_MAF_compensation[5] =  self.right_arm_FT_original_MAF[5] - self.right_arm_force_sensor_data_Moffset[2] - (G[1] * self.right_arm_force_sensor_com[0] - G[0] * self.right_arm_force_sensor_com[1])
  


    def cal_right_arm_FT_original_MAF_compensation_base_coordinate_system(self):     # 将末端六维力传感器的数据从TCP坐标系转换到基坐标系下 现代机器人学
        qpos_ros = self.qpos_lcm2ros(self.joint_current_pos)
        pinocchio.forwardKinematics(self.model, self.data, qpos_ros)

        end_frame_ID_right = self.model.getFrameId("link_ra7")

        target_cart_pose = deepcopy(self.data.oMf[end_frame_ID_right].rotation)

        base_in_effector_pose = target_cart_pose.T
        base_in_effector_0r = np.hstack(((np.zeros((3, 3))), base_in_effector_pose))
        base_in_effector_r0 = np.hstack((base_in_effector_pose, (np.zeros((3, 3)))))
        base_in_effector_ad = np.vstack((base_in_effector_r0, base_in_effector_0r))
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system = base_in_effector_ad.T @ self.right_arm_FT_original_MAF_compensation
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system = 0.01 * self.right_arm_FT_original_MAF_compensation_base_coordinate_system + 0.99 * self.right_arm_FT_original_MAF_compensation_base_coordinate_system_pre
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system_pre = self.right_arm_FT_original_MAF_compensation_base_coordinate_system

