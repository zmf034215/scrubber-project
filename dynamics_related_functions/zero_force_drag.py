from lcm_handler import LCMHandler
import robot_kinematics_and_dynamics_models.dynamic_model
import numpy as np
import time
import csv
import signal
import pinocchio as pin
import os

class Zero_Force_Drag:
    def __init__(self, LCMHandler):
        # lcm
        self.lcm_handler = LCMHandler
        self.interpolation_period = 2
        
        # --- 加载左右臂模型 ---
        path = os.path.dirname(os.path.abspath(__file__))
        # package_dirs = [path + '/..']
        self.model_left = pin.buildModelFromUrdf(path + '/../models/z2_left_arm.urdf')
        self.model_right = pin.buildModelFromUrdf(path + '/../models/z2_right_arm.urdf')
        self.data_left = self.model_left.createData()
        self.data_right = self.model_right.createData()

        self.pre_actual_torq = np.zeros(30)
        self.pre_actual_posi = self.lcm_handler.joint_current_pos
        self.pre_actual_speed = np.zeros(30)
        self.pre_actual_acc = np.zeros(30)
        self.pre_actual_speed_0 = np.zeros(30) 

        self.k_dyn_l_arm = [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0]
        self.k_dyn_r_arm = [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0]
        # self.k_fri_l_arm = [0, 0, 0, 0, 0, 0, 0]
        # self.k_fri_r_arm = [0, 0, 0, 0, 0, 0, 0]
        

        while np.all(self.lcm_handler.joint_current_pos == 0):
            print("等待LCM数据中...")
            time.sleep(0.01)


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

        # 控制状态
        self.recording = False  # 当前是否正在记录
        self.exit_requested = False  # 第三次 Ctrl+C 时触发
        self.csvfile = None
        self.writer = None

        # 用于 Ctrl+C 控制
        self.ctrl_c_count = 0

    def torque_mode_zero_force_drag(self, csv_filename):
        self.csv_filename = csv_filename
        print("✅ 零力拖动启动成功")
        print("👉 按 Ctrl+C 第一次：开始记录")
        print("👉 按 Ctrl+C 第二次：停止记录")
        print("👉 按 Ctrl+C 第三次：退出程序")

        while not self.exit_requested:
            try:
                start_time = time.time()  # 记录循环开始的时间

                self.actual_position_0 = self.lcm_handler.joint_current_pos
                self.actual_speed_0 = self.lcm_handler.joint_current_speed
                self.actual_acc_0 = np.zeros(30) 


                # 对实时位置、速度、加速度滤波
                self.actual_position[:14] = np.array(self.pre_actual_posi[:14]) * (1 - self.alpha_actual) + (self.alpha_actual * self.actual_position_0[:14])
                self.actual_speed[:14] = np.array(self.pre_actual_speed[:14]) * (1 - self.alpha_actual) + (self.alpha_actual * self.actual_speed_0[:14])
                self.actual_acc[:14] = np.array(self.pre_actual_acc[:14]) * (1 - self.alpha_actual) + (self.alpha_actual * self.actual_acc_0[:14])
                
                # --- 拆分左右臂 ---
                q_l, v_l, a_l = self.actual_position[:5], self.actual_speed[:5], self.actual_acc[:5]
                q_r, v_r, a_r = self.actual_position[7:12], self.actual_speed[7:12], self.actual_acc[7:12]

                # --- Pinocchio 计算动力学力矩 ---
                tau_l = np.array(pin.rnea(self.model_left, self.data_left, q_l, v_l, a_l))  #5*1
                tau_r = np.array(pin.rnea(self.model_right, self.data_right, q_r, v_r, a_r))  #5*1
                zeros2 = np.array([0, 0])
                self.dynamic_model_arm_torq = np.concatenate([tau_l, zeros2, tau_r, zeros2])  #14*1
                #  转矩滤波
                self.actual_jointTorqueVec[:14] = np.array(self.pre_actual_torq[:14]) * (1 - self.alpha) + ( self.alpha * self.dynamic_model_arm_torq[:14])
                #  转矩缩放
                self.actual_jointTorqueVec[:14] = (np.array(self.actual_jointTorqueVec[:14]) * (self.k_dyn_l_arm + self.k_dyn_r_arm)).tolist()

                self.lcm_handler.upper_body_data_publisher_torque_mode(self.actual_jointTorqueVec)

                # 数据记录（若已启用记录）
                if self.recording and self.writer:
                    self.writer.writerow(self.actual_position_0[:30].tolist())
                    self.csvfile.flush()

                self.pre_actual_posi = np.copy(self.actual_position)
                self.pre_actual_speed = np.copy(self.actual_speed) 
                self.pre_actual_speed_0 = np.copy(self.actual_speed_0)
                self.pre_actual_acc = np.copy(self.actual_acc) 
                self.pre_actual_torq = np.copy(self.actual_jointTorqueVec)


                # 用于保证下发周期是2ms
                elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
                delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 2毫秒减去已经过的时间
                time.sleep(delay)  # 延迟剩余的时间

            except KeyboardInterrupt:
                self.ctrl_c_count += 1
                if self.ctrl_c_count == 1:
                    print("\n📥 第一次 Ctrl+C：开始记录")
                    if self.csvfile and not self.csvfile.closed:
                        self.csvfile.close()
                    self.csvfile = open(self.csv_filename, 'w', newline='')
                    self.writer = csv.writer(self.csvfile)
                    self.recording = True
                    time.sleep(0.5)
                elif self.ctrl_c_count == 2:
                    print("\n📤 第二次 Ctrl+C：停止记录")
                    self.recording = False
                    if self.csvfile:
                        self.csvfile.close()
                    time.sleep(0.5)
                elif self.ctrl_c_count >= 3:
                    print("\n🛑 第三次 Ctrl+C：退出程序")
                    self.exit_requested = True
                    if self.csvfile and not self.csvfile.closed:
                        self.csvfile.close()

            except Exception as e:
                print(f"程序出现异常: {e}")

    







