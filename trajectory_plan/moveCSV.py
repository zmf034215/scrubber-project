from ..lcm_handler import LCMHandler
import numpy as np
import time
from seven_segment_speed_plan import seven_segment_speed_plan
import csv

class MOVECSV():
    def __init__(self, LCMHandler):
        # lcm
        self.lcm_handler = LCMHandler()

        self.interpolation_result = None

        # csv文件点位下发周期
        self.csv_position_publish_period = 2     # 单位是ms
    

    def movecsvfile(self):
        with open('data.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = (csv.reader(file))

            for row in reader:
                start_time = time.time()  # 记录循环开始的时间
                row = [float(item) for item in row]
                self.interpolation_result = np.array(row)
                cmd_msg = self.convert_to_arm_and_hand_cmd_package_msg(self.interpolation_result)

                # 用于保证下发周期是4ms
                elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
                delay = max(0, self.csv_position_publish_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
                time.sleep(delay)  # 延迟剩余的时间

                self.lcm_handler.upper_body_data_publisher('upper_body_cmd', cmd_msg.encode())

                # # 打印实际循环耗时，用于调试
                # total_elapsed_time = (time.time() - start_time)
                # print("Actual loop time: {:.4f} ms".format(total_elapsed_time * 1000))

            print("CSV文件点位运行结束！！！！")        


