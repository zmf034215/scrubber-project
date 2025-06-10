import pinocchio
import copy
import numpy as np
import time
from lcm_handler import LCMHandler
from trajectory_plan.moveJ import MOVEJ
from trajectory_plan.moveL import MOVEL
from trajectory_plan.moveC import MOVEC
from force_control.force_control_data_cal import Force_Control_Data_Cal
from force_control.force_control import Force_Control

from robot_kinematics_and_dynamics_models.Kinematic_Model import Kinematic_Model

class robot_model():
    def __init__(self):
        ## LCM  
        self.lcm_handler = LCMHandler()

        ## 轨迹规划
        self.movej_plan_target_position_list = None
        self.movel_plan_target_position_list = None
        self.movec_plan_target_position_list = None
        self.trajectory_segment_index = 0

        self.MOVEL = MOVEL(self.lcm_handler)
        self.MOVEJ = MOVEJ(self.lcm_handler)
        self.MOVEC = MOVEC(self.lcm_handler)

        ## 正逆运动学
        self.Kinematic_Model = Kinematic_Model()

        ## 动力学模型


        ## 力控需要的数据处理
        self.Force_Control_Data_Cal = Force_Control_Data_Cal(self.lcm_handler)

        ## 力控
        self.Force_Control = Force_Control(self.lcm_handler, self.Force_Control_Data_Cal)
        


    # 执行该函数之前需要先对movej_plan_target_position_list赋值
    def robot_movej_to_target_position(self):
        for i in range(len(self.movej_plan_target_position_list)):
            with self.lcm_handler.data_lock:
                if(self.trajectory_segment_index == 0):
                    current_joint_position = self.lcm_handler.joint_current_pos.copy()
                    print(current_joint_position)
                else:
                    current_joint_position = self.MOVEJ.interpolation_result

                target_joint_position = self.movej_plan_target_position_list[self.trajectory_segment_index]

                self.MOVEJ.moveJ2target(current_joint_position, target_joint_position)
                self.trajectory_segment_index = self.trajectory_segment_index + 1


    # 执行该函数之前需要先对movel_plan_target_position_list赋值
    def robot_movel_to_target_position(self):
        for i in range(len(self.movel_plan_target_position_list)):
            with self.lcm_handler.data_lock:
                if(self.trajectory_segment_index == 0):
                    current_joint_position = self.lcm_handler.joint_current_pos.copy()
                    print(current_joint_position)
                else:
                    current_joint_position = self.MOVEL.interpolation_result

                target_joint_position = self.movel_plan_target_position_list[self.trajectory_segment_index]

                self.MOVEL.moveL2targetjointposition(current_joint_position, target_joint_position)
                self.trajectory_segment_index = self.trajectory_segment_index + 1

    # 执行该函数之前需要先对movec_plan_target_position_list赋值
    def robot_movec_to_target_position(self):
        with self.lcm_handler.data_lock:
            current_joint_position = self.lcm_handler.joint_current_pos.copy()
            print(current_joint_position)
            middle_joint_position = self.movec_plan_target_position_list[0]
            target_joint_position = self.movec_plan_target_position_list[1]

            self.MOVEC.moveC2target(current_joint_position, middle_joint_position, target_joint_position)