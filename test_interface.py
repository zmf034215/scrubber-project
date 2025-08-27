import argparse
from test_scrubber import test_by_position_control, memory_loop, reset_count
import time
import numpy as np
from robot_model import robot_model
def parse_args():
    """解析命令行参数并返回参数对象"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='测试程序接口，支持从命令行提取参数')
    
    # 添加参数定义
    parser.add_argument('--arm', 
                      type=str, 
                      required=False, 
                      default='right',
                      choices=['left', 'right', 'both'],
                      help='指定机械臂，可选值: left(左臂), right(右臂), both(双臂)')
    
    parser.add_argument('--distance', 
                      type=float, 
                      required=False, 
                      default=0.1,
                      help='指定步长，单位为米，必须是正数')
    
    parser.add_argument('--rotation_deg', 
                      type=float, 
                      required=False, 
                      default=0,
                      help='指定旋转角度，单位为度')
    
    parser.add_argument('--loop', 
                      type=int, 
                      required=False, 
                      default=1,
                      help='指定循环次数，必须是非负整数')
    
    parser.add_argument('--ifreset',
                        type=bool,
                        required=False,
                        default=False,
                        help='是否重置计数器')
    
    parser.add_argument('--rotation_direction',
                        type=int,
                        required=False,
                        default=0,
                        choices=[-1, 0, 1],
                        help='指定旋转方向，-1: 逆时针，0: 保持，1: 顺时针')
    
    parser.add_argument('--x_direction',
                        type=int,
                        required=False,
                        default=0,
                        choices=[-1, 0, 1],
                        help='指定X方向运动方向，-1: 前，0: 保持，1: 后')
    
    parser.add_argument('--y_direction',
                        type=int,
                        required=False,
                        default=1,
                        choices=[-1, 0, 1],
                        help='指定Y方向运动方向，-1: 左，0: 保持，1: 右')
    
    parser.add_argument('--velocity',
                        type=float,
                        required=False,
                        default=0.2,
                        help='指定运动速度，单位为米/秒')
    
    
    # 解析参数
    args = parser.parse_args()
    if args.velocity < 0.2 or args.velocity > 1.2:
        parser.error("速度参数必须在0.2~1.2之间")
    
    if args.rotation_deg > 90 or args.rotation_deg < -90:
        parser.error("旋转角度参数必须在-90度~90度之间")

    return args


if __name__ == '__main__':

    args = parse_args()
    print("\n===== 测试参数 =====")
    print(f"机械臂: {args.arm}")
    print(f"X方向运动方向: {args.x_direction}")
    print(f"Y方向运动方向: {args.y_direction}")
    print(f"移动距离: {args.distance} 米")
    print(f"旋转方向: {args.rotation_direction}")
    print(f"旋转角度: {args.rotation_deg} 度")
    print(f"循环次数: {args.loop}")
    print(f"是否重置计数器: {args.ifreset}")
    print(f"运动速度: {args.velocity} 米/秒")
    print("===================\n")

    
    robot = robot_model()
    time.sleep(1)
    if args.ifreset:
        reset_count()

    # 设置运动控制接口到Kinematic_Model中
    robot.Force_Control.Kinematic_Model.set_robot_interface(robot.MOVEJ, robot.MOVEL, robot.lcm_handler)

    start_pose = np.array([0.3, -0.55, 0.2, -1.8, 0.75, 1.8])  # [x, y, z, roll, pitch, yaw]

    wipe_direction = np.array([args.x_direction, args.y_direction])

    test_by_position_control(robot, args.arm, start_pose, 0.5, wipe_direction=wipe_direction, wipe_total_distance=args.distance, loop=args.loop, rotation_direction=args.rotation_direction, rotation_deg=args.rotation_deg/180*np.pi, velocity=args.velocity)




