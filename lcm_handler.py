from lcm_data_structure.upper_body_cmd_package import upper_body_cmd_package
from lcm_data_structure.upper_body_data_package import upper_body_data_package
from lcm_data_structure.ecat_debug_ft_data_lcmt import ecat_debug_ft_data_lcmt

import lcm
import threading
import numpy as np
import time
from os.path import dirname, join, abspath

import pinocchio

from copy import deepcopy

class LCMHandler:
    def __init__(self):
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

    # 位置模式 传入的参数是期望运行的位置数据
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

    def upper_body_data_publisher_torque_mode(self, torque):
        arm_and_hand_ctrl_msg = upper_body_cmd_package()
        arm_and_hand_ctrl_msg.isUsed = 0
        arm_and_hand_ctrl_msg.control_mode = self.default_control_mode

        ## 一种位置发当前的位置 一种位置发全0
        arm_and_hand_ctrl_msg.jointPosVec = self.joint_current_pos
        # arm_and_hand_ctrl_msg.jointPosVec = np.zeros(30).tolist()

        arm_and_hand_ctrl_msg.jointCurrentVec  = np.zeros(30).tolist()
        arm_and_hand_ctrl_msg.jointSpeedVec = np.zeros(30).tolist()


        # 将整个上半身30维全部的KPKD值都做了修改 此处没有考虑头部以及腰部的情况 后面上整机可能出问题
        arm_and_hand_ctrl_msg.jointKp = (np.zeros(30) + 0.02).tolist()
        arm_and_hand_ctrl_msg.jointKd = (np.zeros(30) + 0.02).tolist()

        arm_and_hand_ctrl_msg.jointTorqueVec = torque

        self.lcm.publish('upper_body_cmd', arm_and_hand_ctrl_msg.encode())
