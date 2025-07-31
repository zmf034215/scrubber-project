import numpy as np
import time
import csv
from lcm_handler import LCMHandler
from trajectory_plan.moveJ import MOVEJ
from trajectory_plan.moveL import MOVEL
from trajectory_plan.moveC import MOVEC
from force_control.force_control_data_cal import Force_Control_Data_Cal
from force_control.force_control import Force_Control

from robot_kinematics_and_dynamics_models.Kinematic_Model import Kinematic_Model
from dynamics_related_functions.zero_force_drag import Zero_Force_Drag
from dynamics_related_functions.collision_detection import Collision_Detection

class robot_model():
    def __init__(self):
        ## LCM  
        self.lcm_handler = LCMHandler()

        ## 正逆运动学
        self.Kinematic_Model = Kinematic_Model()

        ## 动力学相关功能
        self.Zero_Force_Drag = Zero_Force_Drag(self.lcm_handler)
        self.Collision_Detection = Collision_Detection(self.lcm_handler)

        ## 轨迹规划
        self.movej_plan_target_position_list = None
        self.movel_plan_target_position_list = None
        self.movec_plan_target_position_list = None
        self.trajectory_segment_index = 0

        self.MOVEL = MOVEL(self.lcm_handler, self.Collision_Detection)
        self.MOVEJ = MOVEJ(self.lcm_handler, self.Collision_Detection)
        self.MOVEC = MOVEC(self.lcm_handler, self.Collision_Detection)
        self.csv_position_publish_period = 2

        ## 力控需要的数据处理
        self.Force_Control_Data_Cal = Force_Control_Data_Cal(self.lcm_handler, self.Collision_Detection)

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

    
    def get_csv_position_and_interpolation(self):
        with open('offline_say_hi_05_1105_new.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = (csv.reader(file))

            for row in reader:
                start_time = time.time()  # 记录循环开始的时间
                row = [float(item) for item in row]
                interpolation_result = np.array(row)

                self.lcm_handler.upper_body_data_publisher(interpolation_result)

                # 用于保证下发周期是2ms
                elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
                delay = max(0, self.csv_position_publish_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
                time.sleep(delay)  # 延迟剩余的时间
            print("CSV文件点位运行结束！！！！")