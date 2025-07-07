from lcm_handler import LCMHandler
import robot_kinematics_and_dynamics_models.dynamic_model
import numpy as np
import time


class Zero_Force_Drag:
    def __init__(self, LCMHandler):
        # lcm
        self.lcm_handler = LCMHandler
        self.interpolation_period = 2
        self.dynamic_model = robot_kinematics_and_dynamics_models.dynamic_model


        self.pre_actual_torq = np.zeros(30)
        self.pre_actual_posi = np.zeros(30)
        self.pre_actual_speed = np.zeros(30)
        self.pre_actual_acc = np.zeros(30)
        self.pre_actual_speed_0 = np.zeros(30) 

        self.k_dyn_l_arm = [1.0, 0.95, 0.96, 0.8, 0.9, 0.0, 0.0]
        self.k_dyn_r_arm = [0.9, 0.85, 0.5, 0.3, 0.3, 0.0, 0.0]
        self.k_fri_l_arm = [0, 0, 0, 0, 0, 0, 0]
        self.k_fri_r_arm = [0, 0, 0, 0, 0, 0, 0]

        self.alpha = 0.5  # 动力学torq滤波系数
        self.alpha_actual = 0.02    # 实际位置、速度、加速度滤波系数

        #  后面计算各种参数需要的变量
        self.actual_position = np.zeros(30) 
        self.actual_speed = np.zeros(30) 
        self.actual_acc = np.zeros(30) 
        self.actual_jointTorqueVec = np.zeros(30) 
        self.actual_speed_0 = np.zeros(30) 

        self.actual_position_0 = np.zeros(30) 
        self.actual_speed_0 = np.zeros(30) 
        self.actual_acc_0 = np.zeros(30) 

        self.actual_jointTorqueVec = np.zeros(30)
        self.dynamic_model_arm_torq = np.zeros(30)

    def torque_mode_zero_force_drag(self):
        while(1):
            start_time = time.time()  # 记录循环开始的时间

            self.actual_position_0 = self.lcm_handler.joint_current_pos
            self.actual_speed_0 = self.lcm_handler.joint_current_speed
            self.actual_acc_0 = np.zeros(30) 


            # 对实时位置、速度、加速度滤波
            self.actual_position[:14] = np.array(self.pre_actual_posi[:14]) * (1 - self.alpha_actual) + (self.alpha_actual * self.actual_position_0[:14])
            self.actual_speed[:14] = np.array(self.pre_actual_speed[:14]) * (1 - self.alpha_actual) + (self.alpha_actual * self.actual_speed_0[:14])
            self.actual_acc[:14] = np.array(self.pre_actual_acc[:14]) * (1 - self.alpha_actual) + (self.alpha_actual * self.actual_acc_0[:14])

            print("力矩模式-零力拖动！")
            self.dynamic_model_arm_torq = self.dynamic_model.dynamic_cal(self.actual_position[:14], self.actual_speed[:14], self.actual_acc[:14])

            self.actual_jointTorqueVec[:14] = np.array(self.pre_actual_torq[:7]) * (1 - self.alpha) + ( self.alpha * self.dynamic_model_arm_torq[:14])

            self.actual_jointTorqueVec[:14]=(np.array(self.actual_jointTorqueVec[:14]) * (self.k_dyn_l_arm + self.k_dyn_r_arm)).tolist()

            self.lcm_handler.upper_body_data_publisher_torque_mode(self.actual_jointTorqueVec)

            self.pre_actual_posi = np.copy(self.actual_position)
            self.pre_actual_speed = np.copy(self.actual_speed) 
            self.pre_actual_speed_0 = np.copy(self.actual_speed_0)
            self.pre_actual_acc = np.copy(self.actual_acc) 
            self.pre_actual_torq = np.copy(self.actual_jointTorqueVec)


            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 2毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间



    







