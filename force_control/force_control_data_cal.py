from lcm_handler import LCMHandler
import numpy as np
import time
from copy import deepcopy
from robot_kinematics_and_dynamics_models.Kinematic_Model import Kinematic_Model
from trajectory_plan.moveJ import MOVEJ
import threading
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dynamics_related_functions.collision_detection import Collision_Detection



class Force_Control_Data_Cal():
    def __init__(self, LCMHandler, Collision_Detection, Kinematic_Model):
        # LCM
        self.lcm_handler = LCMHandler
        self.Collision_Detection = Collision_Detection
        self.Kinematic_Model = Kinematic_Model()

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

        # 力传感器数据滤波补偿后转换到基坐标系下的结果
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system = np.array([0.0 for i in range(6)])
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system = np.array([0.0 for i in range(6)])

        self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre = np.array([0.0 for i in range(6)])
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system_pre = np.array([0.0 for i in range(6)])       

        self.MOVEJ = MOVEJ(self.lcm_handler, self.Collision_Detection)

        # 传感器数据标定 相关参数设置
        self.hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
        self.hand_home_pos = list(self.hand_home_pos / 180 * np.pi)
        # 此处设置的点位需要包含6个 分别是重力依次位于传感器X轴正方向和负方向 Y轴正方向和负方向 Z轴的正方向和负方向
        self.FT_data_calibration_target_position = [
                                                [0, 0.387,  1.186, -0.584, -1.19,  0,  0, 
                                                 0, -0.387, -1.186, 0.584, 1.19,  0,  0] + self.hand_home_pos + [0, 0, 0, 0],

                                                [-0.3678138256072998, 0.387, 1.569608449935913, -1.2215843200683594, 0.00241919606924057, 0.08925186097621918  + np.pi / 18, -0.04592832922935486 + np.pi / 18, 
                                                 -0.3678138256072998, -0.387, -1.569608449935913, 1.2215843200683594, -0.00241919606924057, 0.08925186097621918  + np.pi / 18, -0.04592832922935486 + np.pi / 18] + self.hand_home_pos + [0, 0, 0, 0],
                                                
                                                [-0.3678138256072998 - np.pi / 2, 0.387, 1.569608449935913, -1.2215843200683594, -1.16796923e+00, 0.08925186097621918 - np.pi / 18, -0.04592832922935486 - np.pi / 18, 
                                                 -0.3678138256072998 - np.pi / 2, -0.387, -1.569608449935913, 1.2215843200683594, 1.16796923e+00, 0.08925186097621918 - np.pi / 18, -0.04592832922935486 - np.pi / 18] + self.hand_home_pos + [0, 0, 0, 0],
                                                
                                                [-2.8 , 0.387, 1.569608449935913, -1.2215843200683594, 0.00241919606924057, 0.08925186097621918, -0.04592832922935486, 
                                                 -2.8 , -0.387, -1.569608449935913, 1.2215843200683594, -0.00241919606924057, 0.08925186097621918, -0.04592832922935486] + self.hand_home_pos + [0, 0, 0, 0],

                                                [-2.8 , 0.387 + np.pi / 2, 1.569608449935913, -1.2215843200683594, -1.16796923e+00, 0.08925186097621918 + np.pi / 18, -0.04592832922935486 - np.pi / 18, 
                                                 -2.8 , -0.387 - np.pi / 2, -1.569608449935913, 1.2215843200683594, 1.16796923e+00, 0.08925186097621918 + np.pi / 18, -0.04592832922935486 - np.pi / 18] + self.hand_home_pos + [0, 0, 0, 0],
                                                
                                                [-0.3678138256072998, 0.387 + np.pi / 2, 1.569608449935913, -1.2215843200683594, 0.00241919606924057, 0.08925186097621918 - np.pi / 18, -0.04592832922935486 + np.pi / 18, 
                                                 -0.3678138256072998, -0.387 - np.pi / 2, -1.569608449935913, 1.2215843200683594, -0.00241919606924057, 0.08925186097621918 - np.pi / 18, -0.04592832922935486 + np.pi / 18] + self.hand_home_pos + [0, 0, 0, 0],

                                            ]
        self.trajectory_segment_index = 0



        ## 传感器标定结果 在更换工装 或者传感器数据不准时 执行FT_data_calibration 把打印的结果替换掉下面的变量
        # self.left_arm_force_sensor_U =  -0.03262756438312473
        # self.left_arm_force_sensor_V =  -0.008130677780862983
        # self.left_arm_force_sensor_G =  5.046246171913459
        # self.left_arm_force_sensor_data_L =  [-0.041007112656776104, 0.16461751075910025, -5.0433936907021515]
        # self.left_arm_force_sensor_com =  [0.013672050589395909, -0.028600226339171392, -0.034997504199793]
        # self.left_arm_force_sensor_data_Foffset =  [3.253067430232532, 2.6557534357589203, 2.0430280661608355]
        # self.left_arm_force_sensor_data_Moffset =  [0.02814232337522351, -0.14687669905210549, 0.13773050274168455]
        # self.left_arm_force_sensor_mass =  0.5149230787666794
        # self.right_arm_force_sensor_U =  -0.0050552972164624975
        # self.right_arm_force_sensor_V =  0.14483456390512195
        # self.right_arm_force_sensor_G =  5.245993771660463
        # self.right_arm_force_sensor_data_L =  [0.7571379319782845, 0.026519944753700897, -5.191000818415489]
        # self.right_arm_force_sensor_com =  [0.030927262600552296, 0.02046379332142403, -0.04045073574611931]
        # self.right_arm_force_sensor_data_Foffset =  [1.9507352952443222, -3.9157230425864413, 3.1944456818709024]
        # self.right_arm_force_sensor_data_Moffset =  [-0.09294613134450672, -0.1554133834741246, -0.15363701753249098]
        # self.right_arm_force_sensor_mass =  0.5353054869041288

        self.left_arm_force_sensor_U =  0.04240215883964988
        self.left_arm_force_sensor_V =  0.06667997350146875
        self.left_arm_force_sensor_G =  4.518628677205159
        self.left_arm_force_sensor_data_L =  [0.30080819355352373, -0.19154220187049165, -4.504534508476377]
        self.left_arm_force_sensor_com =  [0.024225419528576958, -0.02947521701646732, -0.036486426400962016]
        self.left_arm_force_sensor_data_Foffset =  [-0.10526900802226608, -4.212527403004927, 1.5036802451014548]
        self.left_arm_force_sensor_data_Moffset =  [-0.19571002249886676, -0.04026660307583767, -0.09950853380923307]
        self.left_arm_force_sensor_mass =  0.4610845588984856
        self.right_arm_force_sensor_U =  0.052672360450938954
        self.right_arm_force_sensor_V =  -0.03543247256886748
        self.right_arm_force_sensor_G =  3.962800053693389
        self.right_arm_force_sensor_data_L =  [-0.14018773389902922, -0.20863353022447373, -3.9548203138553153]
        self.right_arm_force_sensor_com =  [0.02754084380425616, 0.020178036755968166, -0.03721626056663695]
        self.right_arm_force_sensor_data_Foffset =  [1.7540575084673393, 4.106628964519683, 2.036708726829942]
        self.right_arm_force_sensor_data_Moffset =  [0.19338578261239475, -0.09887502454832245, 0.08129732258104157]
        self.right_arm_force_sensor_mass =  0.4043673524176927

        ## 线程启动要在所有变量声明之后
        self.FT_data_cal_period = 0.002
        self.FT_data_cal_threading_data_lock = threading.Lock()
        self.FT_data_cal_threading_running = True
        self.FT_data_cal_threading = threading.Thread(target = self.FT_data_cal, daemon = True)
        self.FT_data_cal_threading.start()

    def __del__(self):
        self.stop()  # 析构时自动停止线程

    def stop(self):
        self.FT_data_cal_threading_running = False
        if self.FT_data_cal_threading.is_alive():
            self.FT_data_cal_threading.join()



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
            R = (self.Kinematic_Model.left_arm_forward_kinematics(self.lcm_handler.joint_current_pos[:7])).rotation.T
            G = R @ self.left_arm_force_sensor_data_L

            self.left_arm_FT_original_MAF_compensation[0] =  self.left_arm_FT_original_MAF[0] - self.left_arm_force_sensor_data_Foffset[0] - G[0]
            self.left_arm_FT_original_MAF_compensation[1] =  self.left_arm_FT_original_MAF[1] - self.left_arm_force_sensor_data_Foffset[1] - G[1]
            self.left_arm_FT_original_MAF_compensation[2] =  self.left_arm_FT_original_MAF[2] - self.left_arm_force_sensor_data_Foffset[2] - G[2]

            self.left_arm_FT_original_MAF_compensation[3] =  self.left_arm_FT_original_MAF[3] - self.left_arm_force_sensor_data_Moffset[0] - (G[2] * self.left_arm_force_sensor_com[1] - G[1] * self.left_arm_force_sensor_com[2])
            self.left_arm_FT_original_MAF_compensation[4] =  self.left_arm_FT_original_MAF[4] - self.left_arm_force_sensor_data_Moffset[1] - (G[0] * self.left_arm_force_sensor_com[2] - G[2] * self.left_arm_force_sensor_com[0])
            self.left_arm_FT_original_MAF_compensation[5] =  self.left_arm_FT_original_MAF[5] - self.left_arm_force_sensor_data_Moffset[2] - (G[1] * self.left_arm_force_sensor_com[0] - G[0] * self.left_arm_force_sensor_com[1])

    def cal_left_arm_FT_original_MAF_compensation_base_coordinate_system(self):     # 将末端六维力传感器的数据从TCP坐标系转换到基坐标系下 现代机器人学
        target_cart_pose = (self.Kinematic_Model.left_arm_forward_kinematics(self.lcm_handler.joint_current_pos[:7])).rotation
        base_in_effector_pose = target_cart_pose.T
        base_in_effector_0r = np.hstack(((np.zeros((3, 3))), base_in_effector_pose))
        base_in_effector_r0 = np.hstack((base_in_effector_pose, (np.zeros((3, 3)))))
        base_in_effector_ad = np.vstack((base_in_effector_r0, base_in_effector_0r))
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system = base_in_effector_ad.T @ self.left_arm_FT_original_MAF_compensation
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system = 0.01 * self.left_arm_FT_original_MAF_compensation_base_coordinate_system + 0.99 * self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre
        self.left_arm_FT_original_MAF_compensation_base_coordinate_system_pre = self.left_arm_FT_original_MAF_compensation_base_coordinate_system
   
    def update_left_arm_FT_original_buff_and_date_filtering(self):
        # 获取原始FT数据
        raw_ft_data = self.lcm_handler.left_arm_FT_original
        # 处理不同格式的数据
        if isinstance(raw_ft_data, np.ndarray):
            # 如果是二维数组 (6x1)，转换为1维数组
            if raw_ft_data.ndim == 2 and raw_ft_data.shape[1] == 1:
                processed_data = raw_ft_data.flatten().tolist()
            # 如果是1维数组，直接使用
            elif raw_ft_data.ndim == 1:
                processed_data = raw_ft_data.tolist()
            else:
                print(f"警告: 不支持的数组形状 {raw_ft_data.shape}")
                processed_data = [0.0] * 6
        elif isinstance(raw_ft_data, list):
            # 如果是列表的列表，展平为一维列表
            if all(isinstance(item, list) for item in raw_ft_data):
                processed_data = [item[0] if isinstance(item, list) and len(item) > 0 else 0.0 for item in raw_ft_data]
            else:
                processed_data = raw_ft_data
        else:
            print(f"错误: 不支持的数据类型 {type(raw_ft_data)}")
            processed_data = [0.0] * 6
        
        # 确保数据长度为6
        if len(processed_data) != 6:
            print(f"错误: 处理后的FT数据长度应为6, 实际为: {len(processed_data)}")
            processed_data = [0.0] * 6
        
        # 确保所有元素都是数字
        for i, value in enumerate(processed_data):
            if not isinstance(value, (int, float)):
                print(f"错误: 处理后FT数据索引 {i} 不是数字: {type(value)}")
                processed_data = [0.0] * 6
                break

        # 插入新的数据
        self.left_arm_FT_original_buff[1:] = self.left_arm_FT_original_buff[:-1]
        self.left_arm_FT_original_buff[0] = processed_data

        # 均值滤波器
        self.left_arm_FT_original_MAF = self.left_arm_FT_data_moving_average_filter(self.left_arm_FT_original_buff)

        # 传感器数据标定后补偿
        if(self.left_arm_force_sensor_mass != 0):
            self.left_arm_FT_data_compensation()
            self.cal_left_arm_FT_original_MAF_compensation_base_coordinate_system()



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
            R = (self.Kinematic_Model.right_arm_forward_kinematics(self.lcm_handler.joint_current_pos[7:14])).rotation.T  
            G = R @ self.right_arm_force_sensor_data_L

            self.right_arm_FT_original_MAF_compensation[0] =  self.right_arm_FT_original_MAF[0] - self.right_arm_force_sensor_data_Foffset[0] - G[0]
            self.right_arm_FT_original_MAF_compensation[1] =  self.right_arm_FT_original_MAF[1] - self.right_arm_force_sensor_data_Foffset[1] - G[1]
            self.right_arm_FT_original_MAF_compensation[2] =  self.right_arm_FT_original_MAF[2] - self.right_arm_force_sensor_data_Foffset[2] - G[2]

            self.right_arm_FT_original_MAF_compensation[3] =  self.right_arm_FT_original_MAF[3] - self.right_arm_force_sensor_data_Moffset[0] - (G[2] * self.right_arm_force_sensor_com[1] - G[1] * self.right_arm_force_sensor_com[2])
            self.right_arm_FT_original_MAF_compensation[4] =  self.right_arm_FT_original_MAF[4] - self.right_arm_force_sensor_data_Moffset[1] - (G[0] * self.right_arm_force_sensor_com[2] - G[2] * self.right_arm_force_sensor_com[0])
            self.right_arm_FT_original_MAF_compensation[5] =  self.right_arm_FT_original_MAF[5] - self.right_arm_force_sensor_data_Moffset[2] - (G[1] * self.right_arm_force_sensor_com[0] - G[0] * self.right_arm_force_sensor_com[1])        

    def cal_right_arm_FT_original_MAF_compensation_base_coordinate_system(self):    # 将末端六维力传感器的数据从TCP坐标系转换到基坐标系下 现代机器人学
        target_cart_pose = (self.Kinematic_Model.right_arm_forward_kinematics(self.lcm_handler.joint_current_pos[7:14])).rotation
        base_in_effector_pose = target_cart_pose.T
        base_in_effector_0r = np.hstack(((np.zeros((3, 3))), base_in_effector_pose))
        base_in_effector_r0 = np.hstack((base_in_effector_pose, (np.zeros((3, 3)))))
        base_in_effector_ad = np.vstack((base_in_effector_r0, base_in_effector_0r))
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system = base_in_effector_ad.T @ self.right_arm_FT_original_MAF_compensation
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system = 0.01 * self.right_arm_FT_original_MAF_compensation_base_coordinate_system + 0.99 * self.right_arm_FT_original_MAF_compensation_base_coordinate_system_pre
        self.right_arm_FT_original_MAF_compensation_base_coordinate_system_pre = self.right_arm_FT_original_MAF_compensation_base_coordinate_system


    def update_right_arm_FT_original_buff_and_date_filtering(self):

        # 获取原始FT数据
        raw_ft_data = self.lcm_handler.right_arm_FT_original
        # 处理不同格式的数据
        if isinstance(raw_ft_data, np.ndarray):
            # 如果是二维数组 (6x1)，转换为1维数组
            if raw_ft_data.ndim == 2 and raw_ft_data.shape[1] == 1:
                processed_data = raw_ft_data.flatten().tolist()
            # 如果是1维数组，直接使用
            elif raw_ft_data.ndim == 1:
                processed_data = raw_ft_data.tolist()
            else:
                print(f"警告: 不支持的数组形状 {raw_ft_data.shape}")
                processed_data = [0.0] * 6
        elif isinstance(raw_ft_data, list):
            # 如果是列表的列表，展平为一维列表
            if all(isinstance(item, list) for item in raw_ft_data):
                processed_data = [item[0] if isinstance(item, list) and len(item) > 0 else 0.0 for item in raw_ft_data]
            else:
                processed_data = raw_ft_data
        else:
            print(f"错误: 不支持的数据类型 {type(raw_ft_data)}")
            processed_data = [0.0] * 6
        
        # 确保数据长度为6
        if len(processed_data) != 6:
            print(f"错误: 处理后的FT数据长度应为6, 实际为: {len(processed_data)}")
            processed_data = [0.0] * 6
        
        # 确保所有元素都是数字
        for i, value in enumerate(processed_data):
            if not isinstance(value, (int, float)):
                print(f"错误: 处理后FT数据索引 {i} 不是数字: {type(value)}")
                processed_data = [0.0] * 6
                break

        # 插入新的数据
        self.right_arm_FT_original_buff[1:] = self.right_arm_FT_original_buff[:-1]
        self.right_arm_FT_original_buff[0] = processed_data

        # 均值滤波器
        self.right_arm_FT_original_MAF = self.right_arm_FT_data_moving_average_filter(self.right_arm_FT_original_buff)

        # 传感器数据标定后补偿
        if (self.right_arm_force_sensor_mass != 0):
            self.right_arm_FT_data_compensation()
            self.cal_right_arm_FT_original_MAF_compensation_base_coordinate_system()

    def FT_data_cal(self):
        next_time = time.perf_counter()
        while self.FT_data_cal_threading_running:
            next_time += self.FT_data_cal_period
            with self.FT_data_cal_threading_data_lock:
                self.update_left_arm_FT_original_buff_and_date_filtering()
                self.update_right_arm_FT_original_buff_and_date_filtering()

            
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 周期太短来不及睡眠（处理函数太耗时）
                next_time = time.perf_counter()  # 重置时间

    def robot_movej_to_target_position_for_FT_data_calibration(self):
        if (self.trajectory_segment_index == 0):
            current_position = self.lcm_handler.joint_current_pos
        else:
            current_position = self.FT_data_calibration_target_position[self.trajectory_segment_index - 1]

        target_position = self.FT_data_calibration_target_position[self.trajectory_segment_index]

        self.MOVEJ.moveJ2target(current_position, target_position)
        self.trajectory_segment_index = self.trajectory_segment_index + 1


    ## 传感器数据的标定数据计算
    def FT_data_calibration(self):
        Fmat = np.empty((0, 6))
        Mmat = np.empty((0, 1))

        fmat = np.empty((0, 1))
        Rmat = np.empty((0, 6))

        Fmat_right = np.empty((0, 6))
        Mmat_right = np.empty((0, 1))

        fmat_right = np.empty((0, 1))
        Rmat_right = np.empty((0, 6))

        Imat = np.eye(3)

        for i in range (len(self.FT_data_calibration_target_position)):
            self.robot_movej_to_target_position_for_FT_data_calibration()
            time.sleep(1)   

            # 左臂处理
            Rmat_tmp = np.zeros((3, 6))
            fmat_tmp = np.zeros((3, 1))

            R = (self.Kinematic_Model.left_arm_forward_kinematics(self.lcm_handler.joint_current_pos[:7])).rotation.T
            Rmat_tmp = np.hstack((R, Imat))
            fmat_tmp[0] = self.left_arm_FT_original_MAF[0]
            fmat_tmp[1] = self.left_arm_FT_original_MAF[1]
            fmat_tmp[2] = self.left_arm_FT_original_MAF[2]

            Fmat_tmp = np.zeros((3, 6))
            Mmat_tmp = np.zeros((3, 1))

            Fmat_tmp[0, 1] = self.left_arm_FT_original_MAF[2]
            Fmat_tmp[0, 2] = - self.left_arm_FT_original_MAF[1]
            Fmat_tmp[0, 3] = 1

            Fmat_tmp[1, 0] = - self.left_arm_FT_original_MAF[2]
            Fmat_tmp[1, 2] = self.left_arm_FT_original_MAF[0]
            Fmat_tmp[1, 4] = 1

            Fmat_tmp[2, 0] = self.left_arm_FT_original_MAF[1]
            Fmat_tmp[2, 1] = - self.left_arm_FT_original_MAF[0]
            Fmat_tmp[2, 5] = 1

            Mmat_tmp[0] = self.left_arm_FT_original_MAF[3]
            Mmat_tmp[1] = self.left_arm_FT_original_MAF[4]
            Mmat_tmp[2] = self.left_arm_FT_original_MAF[5]

            if i == 0:
                Fmat = Fmat_tmp
                Mmat = Mmat_tmp

                Rmat = Rmat_tmp
                fmat = fmat_tmp

            else:
                Fmat = np.concatenate((Fmat, Fmat_tmp), axis = 0)
                Mmat = np.concatenate((Mmat, Mmat_tmp), axis = 0)

                Rmat = np.concatenate((Rmat, Rmat_tmp), axis = 0)
                fmat = np.concatenate((fmat, fmat_tmp), axis = 0)

            # 右臂处理
            Rmat_tmp_right = np.zeros((3, 6))
            fmat_tmp_right = np.zeros((3, 1))
            R = (self.Kinematic_Model.right_arm_forward_kinematics(self.lcm_handler.joint_current_pos[7:14])).rotation.T

            Rmat_tmp_right = np.hstack((R, Imat))
            fmat_tmp_right[0] = self.right_arm_FT_original_MAF[0]
            fmat_tmp_right[1] = self.right_arm_FT_original_MAF[1]
            fmat_tmp_right[2] = self.right_arm_FT_original_MAF[2]

            Fmat_tmp_right = np.zeros((3, 6))
            Mmat_tmp_right = np.zeros((3, 1))

            Fmat_tmp_right[0, 1] = self.right_arm_FT_original_MAF[2]
            Fmat_tmp_right[0, 2] = - self.right_arm_FT_original_MAF[1]
            Fmat_tmp_right[0, 3] = 1

            Fmat_tmp_right[1, 0] = - self.right_arm_FT_original_MAF[2]
            Fmat_tmp_right[1, 2] = self.right_arm_FT_original_MAF[0]
            Fmat_tmp_right[1, 4] = 1

            Fmat_tmp_right[2, 0] = self.right_arm_FT_original_MAF[1]
            Fmat_tmp_right[2, 1] = - self.right_arm_FT_original_MAF[0]
            Fmat_tmp_right[2, 5] = 1

            Mmat_tmp_right[0] = self.right_arm_FT_original_MAF[3]
            Mmat_tmp_right[1] = self.right_arm_FT_original_MAF[4]
            Mmat_tmp_right[2] = self.right_arm_FT_original_MAF[5]

            if i == 0:
                Fmat_right = Fmat_tmp_right
                Mmat_right = Mmat_tmp_right

                Rmat_right = Rmat_tmp_right
                fmat_right = fmat_tmp_right

            else:
                Fmat_right = np.concatenate((Fmat_right, Fmat_tmp_right), axis = 0)
                Mmat_right = np.concatenate((Mmat_right, Mmat_tmp_right), axis = 0)

                Rmat_right = np.concatenate((Rmat_right, Rmat_tmp_right), axis = 0)
                fmat_right = np.concatenate((fmat_right, fmat_tmp_right), axis = 0)


        # 基于六维力传感器的工业机器人末端负载受力感知研究 -- 张立建  左臂的处理 
        result_tmp = np.linalg.pinv(Fmat.T @ Fmat) @ Fmat.T @ Mmat
        self.left_arm_force_sensor_com = result_tmp[:3, 0] 
        self.left_arm_force_sensor_data_Koffset = result_tmp[3:6, 0] 

        result_tmp = np.linalg.pinv(Rmat.T @ Rmat) @ Rmat.T @ fmat
        self.left_arm_force_sensor_data_L = result_tmp[:3, 0] 
        self.left_arm_force_sensor_data_Foffset = result_tmp[3:6, 0] 

        self.left_arm_force_sensor_G = math.sqrt(self.left_arm_force_sensor_data_L[0] ** 2 + self.left_arm_force_sensor_data_L[1] ** 2 + self.left_arm_force_sensor_data_L[2] ** 2)
        self.left_arm_force_sensor_mass = self.left_arm_force_sensor_G / 9.8
        self.left_arm_force_sensor_V = math.atan(-self.left_arm_force_sensor_data_L[0] / self.left_arm_force_sensor_data_L[2])
        self.left_arm_force_sensor_U = math.asin(-self.left_arm_force_sensor_data_L[1] / self.left_arm_force_sensor_G)

        self.left_arm_force_sensor_data_Moffset[0] = self.left_arm_force_sensor_data_Koffset[0] - self.left_arm_force_sensor_data_Foffset[1] * self.left_arm_force_sensor_com[2] + self.left_arm_force_sensor_data_Foffset[2] * self.left_arm_force_sensor_com[1] 
        self.left_arm_force_sensor_data_Moffset[1] = self.left_arm_force_sensor_data_Koffset[1] - self.left_arm_force_sensor_data_Foffset[2] * self.left_arm_force_sensor_com[0] + self.left_arm_force_sensor_data_Foffset[0] * self.left_arm_force_sensor_com[2] 
        self.left_arm_force_sensor_data_Moffset[2] = self.left_arm_force_sensor_data_Koffset[2] - self.left_arm_force_sensor_data_Foffset[0] * self.left_arm_force_sensor_com[1] + self.left_arm_force_sensor_data_Foffset[1] * self.left_arm_force_sensor_com[0] 

        # 打印传感器数据标定的结果 复制粘贴到初始化的地方 用于传感器示数的补偿
        print(f"self.left_arm_force_sensor_U =  {self.left_arm_force_sensor_U}")
        print(f"self.left_arm_force_sensor_V =  {self.left_arm_force_sensor_V}")
        print(f"self.left_arm_force_sensor_G =  {self.left_arm_force_sensor_G}")
        print(f"self.left_arm_force_sensor_data_L =  {self.left_arm_force_sensor_data_L.tolist()}")
        print(f"self.left_arm_force_sensor_com =  {self.left_arm_force_sensor_com.tolist()}")
        print(f"self.left_arm_force_sensor_data_Foffset =  {self.left_arm_force_sensor_data_Foffset.tolist()}")
        print(f"self.left_arm_force_sensor_data_Moffset =  {self.left_arm_force_sensor_data_Moffset}")
        print(f"self.left_arm_force_sensor_mass =  {self.left_arm_force_sensor_mass}")


        # 基于六维力传感器的工业机器人末端负载受力感知研究 -- 张立建  右臂的处理 
        result_tmp = np.linalg.pinv(Fmat_right.T @ Fmat_right) @ Fmat_right.T @ Mmat_right
        self.right_arm_force_sensor_com = result_tmp[:3, 0] 
        self.right_arm_force_sensor_data_Koffset = result_tmp[3:6, 0] 

        result_tmp = np.linalg.pinv(Rmat_right.T @ Rmat_right) @ Rmat_right.T @ fmat_right
        self.right_arm_force_sensor_data_L = result_tmp[:3, 0] 
        self.right_arm_force_sensor_data_Foffset = result_tmp[3:6, 0] 

        self.right_arm_force_sensor_G = math.sqrt(self.right_arm_force_sensor_data_L[0] ** 2 + self.right_arm_force_sensor_data_L[1] ** 2 + self.right_arm_force_sensor_data_L[2] ** 2)
        self.right_arm_force_sensor_mass = self.right_arm_force_sensor_G / 9.8
        self.right_arm_force_sensor_V = math.atan(-self.right_arm_force_sensor_data_L[0] / self.right_arm_force_sensor_data_L[2])
        self.right_arm_force_sensor_U = math.asin(-self.right_arm_force_sensor_data_L[1] / self.right_arm_force_sensor_G)

        self.right_arm_force_sensor_data_Moffset[0] = self.right_arm_force_sensor_data_Koffset[0] - self.right_arm_force_sensor_data_Foffset[1] * self.right_arm_force_sensor_com[2] + self.right_arm_force_sensor_data_Foffset[2] * self.right_arm_force_sensor_com[1] 
        self.right_arm_force_sensor_data_Moffset[1] = self.right_arm_force_sensor_data_Koffset[1] - self.right_arm_force_sensor_data_Foffset[2] * self.right_arm_force_sensor_com[0] + self.right_arm_force_sensor_data_Foffset[0] * self.right_arm_force_sensor_com[2] 
        self.right_arm_force_sensor_data_Moffset[2] = self.right_arm_force_sensor_data_Koffset[2] - self.right_arm_force_sensor_data_Foffset[0] * self.right_arm_force_sensor_com[1] + self.right_arm_force_sensor_data_Foffset[1] * self.right_arm_force_sensor_com[0] 

        # 打印传感器数据标定的结果 复制粘贴到初始化的地方 用于传感器示数的补偿
        print(f"self.right_arm_force_sensor_U =  {self.right_arm_force_sensor_U}")
        print(f"self.right_arm_force_sensor_V =  {self.right_arm_force_sensor_V}")
        print(f"self.right_arm_force_sensor_G =  {self.right_arm_force_sensor_G}")
        print(f"self.right_arm_force_sensor_data_L =  {self.right_arm_force_sensor_data_L.tolist()}")
        print(f"self.right_arm_force_sensor_com =  {self.right_arm_force_sensor_com.tolist()}")
        print(f"self.right_arm_force_sensor_data_Foffset =  {self.right_arm_force_sensor_data_Foffset.tolist()}")
        print(f"self.right_arm_force_sensor_data_Moffset =  {self.right_arm_force_sensor_data_Moffset}")
        print(f"self.right_arm_force_sensor_mass =  {self.right_arm_force_sensor_mass}")

    def plot_left_arm_FT_original_MAF_compensation_base_coordinate_system(self):
        # 初始化数据存储
        data = np.zeros((6, 50))  # 存储最近50个数据点
        fig, axs = plt.subplots(2, 3, figsize=(25, 15))  # 创建子图，每个方向一个
        fig.suptitle('plot_left_arm_FT_original_MAF_compensation_base_coordinate_system', fontsize=16)
        lines = []  # 存储各曲线线对象
        title = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for i in range(6):
            ax = axs.flatten()[i]
            ax.set_title(title[i])
            ax.set_xlim(0, 50)
            if i < 3:
                ax.set_ylim(-100, 100)  # 假设力的量级在-100到100之间，可以根据实际情况调整
            else:
                ax.set_ylim(-5, 5)  # 假设力的量级在-100到100之间，可以根据实际情况调整
            line, = ax.plot([], [], label=f"Direction {i+1}")
            lines.append(line)
        def update(frame):
            new_data = self.left_arm_FT_original_MAF_compensation_base_coordinate_system # 获取新的传感器数据
            data[:, :-1] = data[:, 1:]  # 数据左移
            data[:, -1] = new_data  # 添加新数据
            
            for i in range(6):
                lines[i].set_data(np.arange(50), data[i, :])  # 更新数据
            return lines

        ani = FuncAnimation(fig, update, frames = np.arange(50), blit=True, repeat=True)
        plt.show()


    def plot_right_arm_FT_original_MAF_compensation_base_coordinate_system(self):
        # 初始化数据存储
        data = np.zeros((6, 50))  # 存储最近50个数据点
        fig, axs = plt.subplots(2, 3, figsize=(25, 15))  # 创建子图，每个方向一个
        fig.suptitle('plot_right_arm_FT_original_MAF_compensation_base_coordinate_system', fontsize=16)
        lines = []  # 存储各曲线线对象
        title = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for i in range(6):
            ax = axs.flatten()[i]
            ax.set_title(title[i])
            ax.set_xlim(0, 50)
            if i < 3:
                ax.set_ylim(-100, 100)  # 假设力的量级在-100到100之间，可以根据实际情况调整
            else:
                ax.set_ylim(-5, 5)  # 假设力的量级在-100到100之间，可以根据实际情况调整
            line, = ax.plot([], [], label=f"Direction {i+1}")
            lines.append(line)
        def update(frame):
            new_data = self.right_arm_FT_original_MAF_compensation_base_coordinate_system # 获取新的传感器数据
            data[:, :-1] = data[:, 1:]  # 数据左移
            data[:, -1] = new_data  # 添加新数据
            
            for i in range(6):
                lines[i].set_data(np.arange(50), data[i, :])  # 更新数据
            return lines

        ani = FuncAnimation(fig, update, frames = np.arange(50), blit=True, repeat=True)
        plt.show()


