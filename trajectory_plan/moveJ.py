from ..lcm_handler import LCMHandler
import numpy as np
import time
from seven_segment_speed_plan import seven_segment_speed_plan
import csv



class MOVEJ():
    def __init__(self, LCMHandler):

        # lcm
        self.lcm_handler = LCMHandler()


        # MOVEJ变量
        self.movej_plan_jerk_max = np.pi * 0.75
        self.movej_plan_acc_max = np.pi * 0.5
        self.movej_plan_speed_max = np.pi / 3
        self.interpolation_period = 2 # 2
        self.joint_position_dim = 30
        self.interpolation_result = np.zeros(self.joint_position_dim)

        self.movej_plan_current_joint_position = None
        self.movej_plan_target_joint_position = None

        self.joint_delta_angle = None
        self.joint_delta_angle_max = None
        self.joint_delta_angle_index = None
        self.joint_movement_direction = None

        self.speed_plan = None
        
        self.MIN_VAL = 0.0000001  

        

    def moveJ2target(self, current_position, target_position):
        current_position = np.array(current_position)
        target_position = np.array(target_position)

        
        self.movej_plan_current_joint_position = current_position
        # print("self.movej_plan_current_joint_position  = {} ".format(self.movej_plan_current_joint_position ))
        self.movej_plan_target_joint_position = target_position
        self.joint_delta_angle = np.zeros(current_position.shape)
        self.joint_movement_direction = np.zeros(current_position.shape)
        for i in range(self.joint_position_dim):
            self.joint_delta_angle[i] = target_position[i] - current_position[i]
            if self.joint_delta_angle[i] > self.MIN_VAL:
                self.joint_movement_direction[i] = 1
            else:
                self.joint_movement_direction[i] = -1
                
            self.joint_delta_angle[i] = np.fabs(self.joint_delta_angle[i])
        
        self.joint_delta_angle_max = np.max(self.joint_delta_angle)
        self.joint_delta_angle_index = np.argmax(self.joint_delta_angle)


        self.speed_plan = seven_segment_speed_plan(self.movej_plan_jerk_max, self.movej_plan_acc_max,
                                                    self.movej_plan_speed_max, self.joint_delta_angle_max)
        self.movej_speed_plan_interpolation()




    def movej_speed_plan_interpolation(self):
        for interpolation_time in np.arange(0, self.speed_plan.time_length, self.interpolation_period / 1000):
            start_time = time.time()  # 记录循环开始的时间
            if 0 <= interpolation_time <= self.speed_plan.accacc_time:
                self.speed_plan.cal_accacc_segment_data(interpolation_time)
            elif self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.uniacc_time + self.speed_plan.accacc_time:
                interpolation_time = interpolation_time - self.speed_plan.accacc_time
                self.speed_plan.cal_uniacc_segment_data(interpolation_time)
            elif self.speed_plan.uniacc_time + self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.acceleration_segment_time:
                interpolation_time = interpolation_time - (self.speed_plan.uniacc_time + self.speed_plan.accacc_time)
                self.speed_plan.cal_decacc_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time:
                interpolation_time = interpolation_time - self.speed_plan.acceleration_segment_time
                self.speed_plan.cal_unispeed_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time:
                interpolation_time = interpolation_time - (self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time)
                self.speed_plan.cal_accdec_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time < interpolation_time <= self.speed_plan.time_length - self.speed_plan.decdec_time:
                interpolation_time = interpolation_time - (self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time)
                self.speed_plan.cal_unidec_segment_data(interpolation_time)
            else:
                interpolation_time = interpolation_time - (self.speed_plan.time_length - self.speed_plan.decdec_time)
                self.speed_plan.cal_decdec_segment_data(interpolation_time)

            for i in range(self.joint_position_dim):
                self.interpolation_result[i] = self.movej_plan_current_joint_position[i] + self.speed_plan.cur_disp_normalization_ratio * self.joint_movement_direction[i] * self.joint_delta_angle[i]

            # print("self.movej_plan_current_joint_position  = {} ".format(self.movej_plan_current_joint_position ))
            # print("self.interpolation_result  = {} ".format(self.interpolation_result ))
            # print("self.movej_plan_target_joint_position  = {} ".format(self.movej_plan_target_joint_position ))


            # with open("interpolate_trajectory.csv", 'a', newline='', encoding='utf-8') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(self.interpolation_result)

            self.lcm_handler.upper_body_data_publisher(self.interpolation_result)

            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间
        
        print("运行结束，到达目标点位！！！")
