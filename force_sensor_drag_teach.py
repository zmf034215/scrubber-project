import time
from robot_model import robot_model


if __name__ == "__main__":
    robot = robot_model()
    time.sleep(1)
    robot.Force_Control.force_sensor_drag_teach_whether_use_IK = True
    robot.Force_Control.force_sensor_drag_teach()
