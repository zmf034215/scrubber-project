#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import lcm

import threading
import os
import sys


sys.path.append("/home/wjy/mycode/arn_control")

from lcm_data_structure.upper_body_cmd_package import upper_body_cmd_package
from lcm_data_structure.upper_body_data_package import upper_body_data_package

class LcmToRos2JointState(Node):
    def __init__(self):
        super().__init__('lcm_to_ros2_joint_state')
        
        # 创建ROS2关节状态发布器
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # 初始化LCM
        self.lcm = lcm.LCM()
        
        # 订阅LCM消息（请替换为实际的LCM主题名称）
        self.lcm.subscribe("upper_body_cmd", self.lcm_callback)
        
        # 启动LCM接收线程
        self.lcm_thread = threading.Thread(target=self.lcm_handle, daemon=True)
        self.lcm_thread.start()
        
        self.get_logger().info("LCM to ROS2 joint state converter node started")
        
        # 关节名称列表（请根据实际机械臂关节名称修改）
        self.joint_names = [
            'joint_la1', 'joint_la2', 'joint_la3', 
            'joint_la4', 'joint_la5', 'joint_la6', 'joint_la7',
            'joint_ra1', 'joint_ra2', 'joint_ra3', 
            'joint_ra4', 'joint_ra5', 'joint_ra6', 'joint_ra7'
        ]

    def lcm_callback(self, channel, data):
        """处理接收到的LCM消息"""
        try:
            # 解析LCM消息
            cmd = upper_body_cmd_package.decode(data)
            
            # 只提取前7个关节角度
            joint_positions = cmd.jointPosVec[:14]
            
            # 创建ROS2关节状态消息
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.name = self.joint_names
            joint_state.position = joint_positions
            # joint_state.position[4] *= -1  
            # joint_state.position[11] *= -1  
            
            # 发布消息
            self.joint_pub.publish(joint_state)
            self.get_logger().debug(f"Published {len(joint_positions)} joint angles")


            # 发布关节状态lcm数据
            data = upper_body_data_package()
            data.curJointPosVec[:14] = joint_positions
            self.lcm.publish("upper_body_data", data.encode())


                
        except Exception as e:
            self.get_logger().error(f"Error processing LCM message: {str(e)}")

    def lcm_handle(self):
        """LCM消息处理循环"""
        while rclpy.ok():
            try:
                self.lcm.handle_timeout(100)  # 100ms超时，避免完全阻塞
            except Exception as e:
                self.get_logger().error(f"LCM handle error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = LcmToRos2JointState()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()