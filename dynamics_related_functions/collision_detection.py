from lcm_handler import LCMHandler
import numpy as np
import time
from copy import deepcopy
import threading


class Collision_Detection():
    def __init__(self, LCMHandler):
        self.lcm_handler = LCMHandler

        self.collision_th = None

        ## 下面的这些参数都需要上真机调试
        self.collision_detection_level = 0                    # 添加碰撞检测等级，等级设置为：0、1、2、3、4、5等级，设置为0时，不启用碰撞检测，等级1→5，碰撞阈值逐步提高
        self.collision_threshold_level_1 = 0.6                # 碰撞检测阈值设定  转矩阈值  
        self.collision_threshold_level_2 = 0.8
        self.collision_threshold_level_3 = 1.0
        self.collision_threshold_level_4 = 1.2
        self.collision_threshold_level_5 = 1.4
        self.joint_torq_rated = [10, 10, 8, 8, 8, 3, 3,     10, 10, 8, 8, 8, 3, 3]             # 设置关节转矩额定值，左右臂共14个关节

        # 碰撞检测标志位 0：没有碰撞 1：发生碰撞
        self.collision_detection_index = 0

        # 循环标志位 用来记录碰撞检测的循环次数
        self.count = 0

        # 碰撞检测实现需要的变量
        self.actual_torq = None
        self.compare_torq = None
        self.pre_actual_torq = None
        self.delta_torq = None
        self.compare_delta_torq = None

        self.collision_detection_cal_threading_start = 0

    def start_collision_detection(self):
        # 判断碰撞和阻抗的启用选择
        if (( self.collision_detection_level != 0 ) & (self.lcm_handler.impedance_control.impedance_control_flag != 0 )) :
            print(f"collision level : {self.collision_detection_level}")
            print(f"impedance_control_flag : {self.lcm_handler.impedance_control.impedance_control_flag}")
            print("collision_detection_level 和 impedance_control.impedance_control_flag 不能同时设置为非零！！！")
            exit()
        self.collision_detection_cal_period = 0.002
        self.collision_detection_cal_threading_data_lock = threading.Lock()
        self.collision_detection_cal_threading_running = True

        # 在开始插补的时候 启动碰撞校验的线程 初始化的时候 先关闭该线程
        self.collision_detection_cal_threading = threading.Thread(target = self.collision_detection, daemon = True)
        self.collision_detection_cal_threading.start()
        self.collision_detection_cal_threading_start = 1

    def __del__(self):
        self.stop_collision_detection()  # 析构时自动停止线程

    def stop_collision_detection(self):
        if(self.collision_detection_cal_threading_start):
            self.collision_detection_cal_threading_running = False
            if self.collision_detection_cal_threading.is_alive():
                self.collision_detection_cal_threading.join()
        else:
            pass

    # 运行之前需要提前在程序中进行碰撞等级的设置 不设置的话默认是不打开碰撞检测的
    def set_collision_detection_level(self):
        if self.collision_detection_level == 0 :
            self.collision_th = 0
        elif self.collision_detection_level == 1 :
            self.collision_th = self.collision_threshold_level_1
        elif self.collision_detection_level == 2 :
            self.collision_th = self.collision_threshold_level_2
        elif self.collision_detection_level == 3 :
            self.collision_th = self.collision_threshold_level_3
        elif self.collision_detection_level == 4 :
            self.collision_th = self.collision_threshold_level_4
        else :
            self.collision_th = self.collision_threshold_level_5

    def collision_detection_cal(self):
        self.count = self.count + 1
        if self.collision_th == 0:
            pass
        else:
            self.actual_torq = self.lcm_handler.joint_current_current_or_torque.copy()
            self.compare_torq  = [abs(a) > abs(b) for a,b in zip(self.actual_torq[:14],self.joint_torq_rated)]

            if self.pre_actual_torq is None:
                if any(self.compare_torq):
                    print("启动时电机转矩超过电机额定值!!!")
                    print("各个关节超出的转矩值为 self.compare_torq = {}".format(self.compare_torq))
                else:
                    pass
            
            else:
                self.delta_torq = [(a - b) for a,b in zip(self.actual_torq[:14], self.pre_actual_torq[:14])]
                self.compare_delta_torq = [abs(a) > self.collision_th for a in self.delta_torq]  
                if any(self.compare_delta_torq): 
                    print("当前阈值设置下，发生了碰撞！！！")
                    self.collision_detection_index = 1
                    print("各个关节超出的阈值为 self.delta_torq = {}".format(self.delta_torq))
                    print(self.compare_delta_torq)

                elif any(self.compare_torq):
                    print(self.compare_torq)
                    self.collision_detection_index = 1
                    print("当前电流值超过了额定电流值！！！")
                else:
                    pass

            if self.count < 8:
                self.pre_actual_torq = None
            else:
                self.pre_actual_torq = self.actual_torq


    def collision_detection(self):
        next_time = time.perf_counter()
        self.set_collision_detection_level()
        while self.collision_detection_cal_threading_running:
            next_time += self.collision_detection_cal_period
            with self.collision_detection_cal_threading_data_lock:
                # 碰撞检测具体实现函数
                self.collision_detection_cal()

            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 周期太短来不及睡眠（处理函数太耗时）
                next_time = time.perf_counter()  # 重置时间
