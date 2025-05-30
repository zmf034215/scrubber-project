import time
from robot_model import robot_model


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)
    # 力传感器数据的标定 标定程序只有在重新装配传感器 或者传感器数据误差大 或者安装的工装更换时 进行标定
    # 标定结束后 结果打印在终端 复制到初始化对应的部分 便可正常完成数据的补偿 力传感器补偿后的数据变量是  self.FT_original_MAF_compensation 后面做力控也应该用这个变量
    robot.Force_Control_Data_Cal.FT_data_calibration()
    robot.Force_Control_Data_Cal.stop()
