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
from hybrid_force_and_pos_control.hybrid_force_movel import Hybrid_Force_MoveL
from hybrid_force_and_pos_control.hybrid_force_movec import Hybrid_Force_MoveC

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

        ## 力控需要的数据处理
        self.Force_Control_Data_Cal = Force_Control_Data_Cal(self.lcm_handler, self.Collision_Detection, self.Kinematic_Model)

        ## 力控
        self.Force_Control = Force_Control(self.lcm_handler, self.Force_Control_Data_Cal, self.Kinematic_Model)

        self.MOVEL = MOVEL(self.lcm_handler, self.Collision_Detection, self.Kinematic_Model, self.Force_Control)
        self.MOVEJ = MOVEJ(self.lcm_handler, self.Collision_Detection)
        self.MOVEC = MOVEC(self.lcm_handler, self.Collision_Detection,self.Kinematic_Model)
        self.csv_position_publish_period = 2

        ## 力控需要的数据处理
        self.Force_Control_Data_Cal = Force_Control_Data_Cal(self.lcm_handler)

        ## 力控
        self.Force_Control = Force_Control(self.lcm_handler, self.Force_Control_Data_Cal)

        ## 力位混合控制
        self.Hybrid_Force_MoveL = Hybrid_Force_MoveL(self.lcm_handler, self.Collision_Detection, self.Force_Control_Data_Cal)
        self.Hybrid_Force_MoveC = Hybrid_Force_MoveC(self.lcm_handler, self.Collision_Detection, self.Force_Control_Data_Cal)

        ## 基于笛卡尔空间的力位混合控制输入
        ## 输入SE3元素的列表
        self.hybrid_force_movel_plan_left_target_cart_list = None
        self.hybrid_force_movel_plan_right_target_cart_list = None

        self.hybrid_force_movec_plan_left_target_cart_list = None
        self.hybrid_force_movec_plan_right_target_cart_list = None
        
        ## 基于关节空间的输入
        self.hybrid_force_movel_plan_target_joint_list = None
        self.hybrid_force_movec_plan_target_joint_list = None

        ## 
        self.hybrid_force_movel_plan_target_FT_data_list = None
        self.hybrid_force_movec_plan_target_FT_data_list = None
        
        ## 设置力的阈值
        self.hybrid_force_threshold = 0

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

    # 基于笛卡尔空间的混合力movel
    # 执行该函数之前需要先对hybrid_force_movel_plan_target_position_list和hybrid_force_movel_plan_target_FT_data_list赋值
    def robot_hybrid_force_movel_to_target_cart(self, threshold = 0):
        self.hybrid_force_threshold = threshold
        target_FT_data = self.hybrid_force_movel_plan_target_FT_data_list[0]
        # print(target_FT_data)

        if threshold <= 1 and threshold > 0 and np.linalg.norm(np.array(target_FT_data)) > 0:
            # 如果设计了力执行阈值
                middle_FT_data = target_FT_data * threshold
                # 设置纯导纳驱动的中间力
                self.Force_Control.left_arm_target_FT_data = middle_FT_data[:6]
                self.Force_Control.right_arm_target_FT_data = middle_FT_data[6:]
                # 先在力方向上执行纯力导纳控制到指定的力阈值
                self.Force_Control.constant_force_tracking_control()
                
        for i in range(len(self.hybrid_force_movel_plan_left_target_cart_list)):
            with self.lcm_handler.data_lock:
                if(self.trajectory_segment_index == 0):
                    hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
                    hand_home_pos = list(hand_home_pos / 180 * np.pi)
                    current_joint_position  = np.array([-0.3, 0.7, 1.5, -1.27, -2.2, 0.2, 0,
                                                 -0.3, -0.7, -1.5, 1.27, 2.2, -0.2, 0] + hand_home_pos + [0, 0, 0, 0])
                    # current_joint_position = self.lcm_handler.joint_current_pos.copy()
                    left_current_cart_position = self.Kinematic_Model.left_arm_forward_kinematics(current_joint_position[:7])
                    right_current_cart_position = self.Kinematic_Model.right_arm_forward_kinematics(current_joint_position[7:14])
                    print(left_current_cart_position)
                    print(right_current_cart_position)
                else:
                    current_joint_position = self.Hybrid_Force_MoveL.interpolation_result
                    # print(current_joint_position)
                    left_current_cart_position = self.Kinematic_Model.left_arm_forward_kinematics(current_joint_position[:7])
                    right_current_cart_position = self.Kinematic_Model.right_arm_forward_kinematics(current_joint_position[7:14])

                left_target_cart_position = self.hybrid_force_movel_plan_left_target_cart_list[self.trajectory_segment_index]
                right_target_cart_position = self.hybrid_force_movel_plan_right_target_cart_list[self.trajectory_segment_index]

                target_FT_data = self.hybrid_force_movel_plan_target_FT_data_list[self.trajectory_segment_index]
                self.Hybrid_Force_MoveL.robot_hybrid_force_movel_by_cart(left_current_cart_position, right_current_cart_position, left_target_cart_position, 
                                                                   right_target_cart_position, target_FT_data)

                self.trajectory_segment_index = self.trajectory_segment_index + 1

    # 基于笛卡尔空间的混合力movec
    # 执行该函数之前需要先对hybrid_force_movel_plan_target_position_list和hybrid_force_movel_plan_target_FT_data_list赋值
    def robot_hybrid_force_movec_to_target_cart(self, threshold = 0):
        self.hybrid_force_threshold = threshold
        target_FT_data = self.hybrid_force_movec_plan_target_FT_data_list[0]

        with self.lcm_handler.data_lock:
            if threshold <= 1 and threshold > 0 and np.linalg.norm(np.arrar(target_FT_data)) > 0:
            # 如果设计了力执行阈值
                middle_FT_data = target_FT_data * threshold
                # 设置纯导纳驱动的中间力
                self.Force_Control.left_arm_target_FT_data = middle_FT_data[:6]
                self.Force_Control.right_arm_target_FT_data = middle_FT_data[6:]
                # 先在力方向上执行纯力导纳控制到指定的力阈值
                self.Force_Control.constant_force_tracking_control()
            
            current_joint_position = self.lcm_handler.joint_current_pos.copy()
            left_current_cart_position = self.Kinematic_Model.left_arm_forward_kinematics(current_joint_position[:7])
            right_current_cart_position = self.Kinematic_Model.right_arm_forward_kinematics(current_joint_position[7:14])
            print(left_current_cart_position)
            print(right_current_cart_position)

            left_middle_cart_position = self.hybrid_force_movec_plan_left_target_cart_list[0]
            left_target_cart_position = self.hybrid_force_movec_plan_left_target_cart_list[1]


            right_middle_cart_position = self.hybrid_force_movec_plan_right_target_cart_list[0]
            right_target_cart_position = self.hybrid_force_movec_plan_right_target_cart_list[1]

            

            self.Hybrid_Force_MoveC.hybrid_force_movec_control_by_cart(left_current_cart_position, left_middle_cart_position, left_target_cart_position, 
                                                                       right_current_cart_position, right_middle_cart_position, right_target_cart_position,
                                                                        target_FT_data, self.hybrid_force_threshold)
    # 基于关节的混合力movel
    def robot_hybrid_force_movel_to_target_joint(self, threshold = 0):
        self.hybrid_force_threshold = threshold
        target_FT_data = self.hybrid_force_movel_plan_target_FT_data_list[0]
        if threshold <= 1 and threshold > 0 and np.linalg.norm(np.arrar(target_FT_data)) > 0:
            # 如果设计了力执行阈值
                middle_FT_data = target_FT_data * threshold
                # 设置纯导纳驱动的中间力
                self.Force_Control.left_arm_target_FT_data = middle_FT_data[:6]
                self.Force_Control.right_arm_target_FT_data = middle_FT_data[6:]
                # 先在力方向上执行纯力导纳控制到指定的力阈值
                self.Force_Control.constant_force_tracking_control()

        for i in range(len(self.hybrid_force_movel_plan_target_joint_list)):
            with self.lcm_handler.data_lock:
                if(self.trajectory_segment_index == 0):
                    current_joint_position = self.lcm_handler.joint_current_pos.copy()
                    print(current_joint_position)
                else:
                    current_joint_position = self.Hybrid_Force_MoveL.interpolation_result
                    
                target_joint_position = self.hybrid_force_movel_plan_target_joint_list[self.trajectory_segment_index]
                target_FT_data = self.hybrid_force_movel_plan_target_FT_data_list[self.trajectory_segment_index]
                self.Hybrid_Force_MoveL.robot_hybrid_force_movel_by_joint(current_joint_position, target_joint_position, target_FT_data, self.hybrid_force_threshold)

                self.trajectory_segment_index = self.trajectory_segment_index + 1

   

    
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