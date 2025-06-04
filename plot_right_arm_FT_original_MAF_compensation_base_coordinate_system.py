import time
from robot_model import robot_model


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)
    robot.Force_Control_Data_Cal.plot_right_arm_FT_original_MAF_compensation_base_coordinate_system()
