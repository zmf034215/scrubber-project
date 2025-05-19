import lcm
import threading
import time

from lcm_data_structure.lcm_command_struct import lcm_command_struct
from lcm_data_structure.lcm_response_lcmt import lcm_response_lcmt

class RobotFSMController():
    def __init__(self) -> None:
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=1')
        self.lcm_thread_handle = threading.Thread(target = self.lcm_handle, daemon = True)
        self.lcm.subscribe('lcm_response', self.get_response_from_robot)
        self.lcm_thread_handle.start()
        self.current_fsm = None
        self.current_switch_flag = None

    def lcm_handle(self):
        '''
        block func
        '''
        while True:
            self.lcm.handle()
    
    def get_response_from_robot(self, channel, data):
        try:
            msg = lcm_response_lcmt.decode(data)
            self.current_fsm = msg.robot_fsm
            self.current_switch_flag = msg.switch_flag
        except Exception as e:
            print("Failed to decode message or update variables:", e)

    def set_cmd_to_lcm_command_struct(self, fsm):
        cmd_struct = lcm_command_struct()
        cmd_struct.robot_fsm = fsm
        cmd_struct.x_vel_des = 0
        cmd_struct.y_vel_des = 0
        cmd_struct.yaw_vel_des = 0
        cmd_struct.rpy_des = [0, 0, 0]
        cmd_struct.stop = 0
        return cmd_struct

    def set_robot_fsm(self, fsm):
        cmd_msg = self.set_cmd_to_lcm_command_struct(fsm)
        while True:
            self.lcm.publish('lcm_com_cmd', cmd_msg.encode())
            print("set robot fsm to: {}".format(fsm))
            if self.current_fsm == fsm:
                print("set robot fsm to: {} success".format(fsm))
                break



if __name__ == "__main__":
    controller = RobotFSMController()
    controller.set_robot_fsm(45)
