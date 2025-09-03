import numpy as np
import time
import csv
from scipy.interpolate import interp1d
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
        self.MOVEC = MOVEC(self.lcm_handler, self.Collision_Detection, self.Kinematic_Model)
        self.csv_position_publish_period = 2

        ## 力位混合控制
        self.Hybrid_Force_MoveL = Hybrid_Force_MoveL(self.lcm_handler, self.Collision_Detection, self.Force_Control_Data_Cal)

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
        target_FT_data = np.array(target_FT_data)
        # print(target_FT_data)

        if threshold <= 1 and threshold > 0 and np.linalg.norm(target_FT_data) > 0:
            # 如果设计了力执行阈值
                middle_FT_data = target_FT_data * threshold
                # 设置纯导纳驱动的中间力
                self.Force_Control.left_arm_target_FT_data = middle_FT_data[:6]
                self.Force_Control.right_arm_target_FT_data = middle_FT_data[6:]
                # 先在力方向上执行纯力导纳控制到指定的力阈值
                self.Force_Control.constant_force_tracking_control(flag=1)
        
        print("开始力位混合控制过程")
        for i in range(len(self.hybrid_force_movel_plan_left_target_cart_list)):
            with self.lcm_handler.data_lock:
                if(self.trajectory_segment_index == 0):

                    current_joint_position = self.lcm_handler.joint_current_pos.copy()
                    left_current_cart_position = self.Kinematic_Model.left_arm_forward_kinematics(current_joint_position[:7])
                    right_current_cart_position = self.Kinematic_Model.right_arm_forward_kinematics(current_joint_position[7:14])

                else:
                    current_joint_position = self.Hybrid_Force_MoveL.interpolation_result
                    # print(current_joint_position)
                    left_current_cart_position = self.Kinematic_Model.left_arm_forward_kinematics(current_joint_position[:7])
                    right_current_cart_position = self.Kinematic_Model.right_arm_forward_kinematics(current_joint_position[7:14])

                left_target_cart_position = self.hybrid_force_movel_plan_left_target_cart_list[self.trajectory_segment_index]
                right_target_cart_position = self.hybrid_force_movel_plan_right_target_cart_list[self.trajectory_segment_index]

                # target_FT_data = self.hybrid_force_movel_plan_target_FT_data_list[self.trajectory_segment_index]
                self.Hybrid_Force_MoveL.robot_hybrid_force_movel_by_cart(left_current_cart_position, right_current_cart_position, left_target_cart_position, 
                                                                   right_target_cart_position, target_FT_data)

                self.trajectory_segment_index = self.trajectory_segment_index + 1

   
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

    
    def get_csv_position_and_interpolation(self, csv_filename, speed_scale):
        # Step 1: 读取CSV所有点位到列表中
        with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
            reader = (csv.reader(file))
            reader = (csv.reader(file))
            reader = list(reader)  # 读取所有数据为列表
            reader = self.resample_csv_trajectory(reader,speed_scale)  # 通过speed_scale的设置来对csv文件中的轨迹进行调速
            csv_positions = [np.array([float(item) for item in row]) for row in reader]

        if not csv_positions:
            print("CSV文件为空！")
            return
        
        # Step 3: 设置MOVEJ的目标列表为 [当前点, csv第一个点]
        self.movej_plan_target_position_list = [csv_positions[0]]
        self.trajectory_segment_index = 0  # 重置轨迹段索引
        self.robot_movej_to_target_position()  # 运动到CSV的起始点

        # Step 4: 等待稳定一段时间（可选）
        time.sleep(0.5)

        # Step 5: 按2ms周期开始发布CSV数据
        for row in csv_positions:
            start_time = time.time()  # 记录循环开始的时间
            
            self.lcm_handler.upper_body_data_publisher(row)

            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.csv_position_publish_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间
        print("CSV文件点位运行结束！！！！")

    def resample_csv_trajectory(self, points: np.ndarray, speed_scale: float):   # 通过speed_scale的设置来对csv文件中的轨迹进行调速
        """
        根据 speed_scale 对轨迹点进行插值或抽稀
        - speed_scale > 1 → 抽稀 → 快
        - speed_scale < 1 → 插值 → 慢
        """
        # from scipy.interpolate import interp1d
        assert speed_scale > 0, "speed_scale 必须大于 0"
        original_len = len(points)
        new_len = int(original_len / speed_scale)

        if new_len < 2:
            raise ValueError("调整后的轨迹点太少，speed_scale 太大")
    
        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, new_len)

        interpolated = interp1d(x_old, points, axis=0)(x_new)
        return interpolated
    