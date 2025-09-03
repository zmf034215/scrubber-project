import os
import pinocchio
import numpy as np
from numpy.linalg import norm, pinv
import time 
import copy


class Kinematic_Model:
    def __init__(self):

        parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(parent_folder, "models", "z2_left_arm.urdf")
        self.left_arm_pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.left_arm_pin_data = self.left_arm_pin_model.createData()
        print('model name: ' + self.left_arm_pin_model.name)


        urdf_path = os.path.join(parent_folder, "models", "z2_right_arm.urdf")
        self.right_arm_pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.right_arm_pin_data = self.right_arm_pin_model.createData()
        print('model name: ' + self.right_arm_pin_model.name)

        self.left_arm_interpolation_result = np.zeros(7)
        self.right_arm_interpolation_result = np.zeros(7)

        self.left_arm_inverse_kinematics_solution_success_flag = None
        self.right_arm_inverse_kinematics_solution_success_flag = None

               

    def left_arm_forward_kinematics(self, left_arm_joint_position):
        pinocchio.forwardKinematics(self.left_arm_pin_model, self.left_arm_pin_data, left_arm_joint_position[:5])
        pinocchio.updateFramePlacements(self.left_arm_pin_model, self.left_arm_pin_data)
        frame_id = self.left_arm_pin_model.getFrameId("link_la5")
        left_arm_cart_pose = copy.deepcopy(self.left_arm_pin_data.oMf[frame_id])
        return left_arm_cart_pose
    
    def left_arm_Jacobians(self, left_arm_joint_position):
        pinocchio.forwardKinematics(self.left_arm_pin_model, self.left_arm_pin_data, left_arm_joint_position[:5])
        pinocchio.updateFramePlacements(self.left_arm_pin_model, self.left_arm_pin_data) 
        pinocchio.computeJointJacobians(self.left_arm_pin_model, self.left_arm_pin_data) 
        
        frame_id = self.left_arm_pin_model.getFrameId("link_la5")
        Jacobians = pinocchio.getFrameJacobian(self.left_arm_pin_model, self.left_arm_pin_data, frame_id, pinocchio.LOCAL)
        return Jacobians      


    def right_arm_forward_kinematics(self, right_arm_joint_position):
        pinocchio.forwardKinematics(self.right_arm_pin_model, self.right_arm_pin_data, right_arm_joint_position[:5])
        pinocchio.updateFramePlacements(self.right_arm_pin_model, self.right_arm_pin_data)
        frame_id = self.right_arm_pin_model.getFrameId("link_ra5")
        right_arm_cart_pose = copy.deepcopy(self.right_arm_pin_data.oMf[frame_id])
        return right_arm_cart_pose
    
    def right_arm_Jacobians(self, right_arm_joint_position):
        pinocchio.forwardKinematics(self.right_arm_pin_model, self.right_arm_pin_data, right_arm_joint_position[:5])
        pinocchio.updateFramePlacements(self.right_arm_pin_model, self.right_arm_pin_data) 
        pinocchio.computeJointJacobians(self.right_arm_pin_model, self.right_arm_pin_data)     

        frame_id = self.right_arm_pin_model.getFrameId("link_ra5")
        Jacobians = pinocchio.getFrameJacobian(self.right_arm_pin_model, self.right_arm_pin_data, frame_id, pinocchio.LOCAL)
        return Jacobians      

    def left_arm_inverse_kinematics(self, cart_interpolation_pose, cart_interpolation_position, current_joint_position):
        oMdes = pinocchio.SE3(np.array(cart_interpolation_pose), np.array(cart_interpolation_position))
        eps    = 1e-7
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-5
        q = current_joint_position[:5]
        i = 1 

        frame_id = self.left_arm_pin_model.getFrameId("link_la5")
        
        while(1):
            pinocchio.forwardKinematics(self.left_arm_pin_model, self.left_arm_pin_data, q[:5])
            pinocchio.updateFramePlacements(self.left_arm_pin_model, self.left_arm_pin_data) 
            iMd = self.left_arm_pin_data.oMf[frame_id].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame
            if np.linalg.norm(err) < eps:
                self.left_arm_interpolation_result = q
                self.left_arm_inverse_kinematics_solution_success_flag = True
                self.left_arm_interpolation_result = np.array((self.left_arm_interpolation_result).tolist() + [0, 0])
                # print("逆解循环了:{} 次".format(i))
                break
            if i >= IT_MAX:
                self.left_arm_inverse_kinematics_solution_success_flag = False
                self.left_arm_interpolation_result = current_joint_position
                break
            
            J = self.left_arm_Jacobians(q)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            # v = -np.linalg.solve(J.T.dot(J) + damp * np.eye(7), J.T.dot(err))
            q = pinocchio.integrate(self.left_arm_pin_model, q, v * DT)

            i += 1

    def right_arm_inverse_kinematics(self, cart_interpolation_pose, cart_interpolation_position, current_joint_position):
        oMdes = pinocchio.SE3(np.array(cart_interpolation_pose), np.array(cart_interpolation_position))
        eps    = 1e-7
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-12
        q = current_joint_position[:5]
        i = 1 

        frame_id = self.right_arm_pin_model.getFrameId("link_ra5")

        while(1):
            pinocchio.forwardKinematics(self.right_arm_pin_model, self.right_arm_pin_data, q[:5])
            pinocchio.updateFramePlacements(self.right_arm_pin_model, self.right_arm_pin_data) 
            iMd = self.right_arm_pin_data.oMf[frame_id].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame
            if np.linalg.norm(err) < eps:
                self.right_arm_interpolation_result = q # 5*1
                self.right_arm_inverse_kinematics_solution_success_flag = True
                self.right_arm_interpolation_result = np.array((self.right_arm_interpolation_result).tolist() + [0, 0])
                
                break
            if i >= IT_MAX:
                self.right_arm_inverse_kinematics_solution_success_flag = False
                self.right_arm_interpolation_result = current_joint_position
                print(f"i = {i}")
                break

            # pinocchio.computeJointJacobians(self.right_arm_pin_model, self.right_arm_pin_data)
            # J = pinocchio.getJointJacobian(self.right_arm_pin_model, self.right_arm_pin_data, self.right_arm_pin_model.njoints - 1, pinocchio.LOCAL)
            J = self.right_arm_Jacobians(q)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.right_arm_pin_model, q, v * DT)

            i += 1

    def set_robot_interface(self, movej_obj, movel_obj, lcm_handler):
        """
        设置上层机器人控制接口
        """
        self.MOVEJ = movej_obj
        self.MOVEL = movel_obj
        self.lcm_handler = lcm_handler
        self.interpolation_period = 2  # 默认控制周期为 2ms

    def move_to_start_pose(self, arm, start_position):
        """
        使用 MOVEL 运动到指定位姿（通过逆解 + 关节空间L轨迹）
        cartesian_pose: [x, y, z, roll, pitch, yaw]
        """
        if self.MOVEL is None:
            print("❌ 未设置 MOVEL 控制器")
            return False
        
        hand_home_pos = np.array([165, 176, 176, 176, 25.0, 165.0, 165, 176, 176, 176, 25.0, 165.0],dtype = np.float64)
        hand_home_pos = list(hand_home_pos / 180 * np.pi)
        prepare_joint_posi1 = [0.3,  0.2,  0.0, -0.3,  0.0, 0, 0,
                               0.3, -0.2, -0.0, -0.3, -0.0, 0, 0] + hand_home_pos + [0, 0, 0, 0]
        prepare_joint_posi2 = [0,  0.2,  0.5, 0.0,  0.0, 0, 0,
                               0, -0.2, -0.5, 0.0, -0.0, 0, 0] + hand_home_pos + [0, 0, 0, 0]
        target_joint_position = start_position  + hand_home_pos + [0, 0, 0, 0]

        current_joint_position = self.lcm_handler.joint_current_pos.copy()   # 30*1
        
        self.MOVEJ.moveJ2target(current_joint_position, prepare_joint_posi1)
        time.sleep(0.1)
        self.MOVEJ.moveJ2target( prepare_joint_posi1, prepare_joint_posi2)
        time.sleep(0.1)

        # # 构造目标位姿
        # xyz = cartesian_pose[:3]
        # rpy = cartesian_pose[3:]
        # try:
        #     R = pinocchio.rpy.rpyToMatrix(rpy)
        # except Exception as e:
        #     print(f"❌ 姿态转换失败: {e}")
        #     return False

        # # 求逆解
        # if arm == 'right':
        #     self.right_arm_inverse_kinematics(R, xyz, np.array(prepare_joint_posi2[7:14]))
        #     if self.right_arm_inverse_kinematics_solution_success_flag:
        #         target_joint_position_r = self.right_arm_interpolation_result
        #         target_joint_position =np.array([-0.1,  0.2,  0.5, 0.3,  0.5, 0, 0] + target_joint_position_r.tolist() + prepare_joint_posi2[14:])
        #     else:
        #         print("❌ 右臂逆解失败")
        #         return False
        # else:
        #     self.left_arm_inverse_kinematics(R, xyz, np.array(prepare_joint_posi2[:7]))
        #     if self.left_arm_inverse_kinematics_solution_success_flag:
        #         target_joint_position_l = self.left_arm_interpolation_result
        #         target_joint_position =np.array(target_joint_position_l.tolist() + [-0.3, -0.2, -0.5, 0.6, -0.5, 0, 0] + prepare_joint_posi2[14:])
        #     else:
        #         print("❌ 左臂逆解失败")
        #         return False
        
        # self.MOVEL.moveL2targetjointposition(prepare_joint_posi2, target_joint_position)
        self.MOVEJ.moveJ2target(prepare_joint_posi2, target_joint_position)
        return True

    def move_relative(self, arm, delta_xyz):
        """
        相对当前位置移动 delta_xyz（保持姿态）
        """
        if self.MOVEL is None:
            print("❌ MOVEL 未设置")
            return False

        current_joint_position = self.lcm_handler.joint_current_pos.copy()

        # 当前末端位姿
        if arm == 'right':
            current_pose = self.right_arm_forward_kinematics(current_joint_position[7:14])
        else:
            current_pose = self.left_arm_forward_kinematics(current_joint_position[:7])

        current_position = current_pose.translation
        print("current_posi :",current_position)
        target_position = current_position + delta_xyz
        print("target_posi :",target_position)
        R = current_pose.rotation  # 保持姿态

        # 求逆解
        if arm == 'right':
            self.right_arm_inverse_kinematics(R, target_position, current_joint_position[7:14])
            if self.right_arm_inverse_kinematics_solution_success_flag:
                target_joint_position_r = self.right_arm_interpolation_result
                target_joint_position =np.array(current_joint_position[:7].tolist() + target_joint_position_r.tolist() + current_joint_position[14:].tolist())
            else:
                print("❌ 右臂逆解失败!!!")
                return False
        else:
            self.left_arm_inverse_kinematics(R, target_position, current_joint_position[:7])
            if self.left_arm_inverse_kinematics_solution_success_flag:
                target_joint_position_l = self.left_arm_interpolation_result
                target_joint_position =np.array(target_joint_position_l.tolist() + current_joint_position[7:].tolist())
            else:
                print("❌ 左臂逆解失败")
                return False

        # 执行相对运动
        self.MOVEL.moveL2targetjointposition(current_joint_position, target_joint_position)
        return True

    def back_to_start_pose(self, arm, cartesian_pose):
        """
        使用 MOVEL 运动到指定位姿（通过逆解 + 关节空间L轨迹）
        cartesian_pose: [x, y, z, roll, pitch, yaw]
        """
        if self.MOVEL is None:
            print("❌ 未设置 MOVEL 控制器")
            return False
                
        current_joint_position = self.lcm_handler.joint_current_pos.copy()
                
        # 构造目标位姿
        xyz = cartesian_pose[:3]
        rpy = cartesian_pose[3:]
        try:
            R = pinocchio.rpy.rpyToMatrix(rpy)
        except Exception as e:
            print(f"❌ 姿态转换失败: {e}")
            return False

        # 求逆解
        if arm == 'right':
            self.right_arm_inverse_kinematics(R, xyz, np.array(current_joint_position[7:14]))
            if self.right_arm_inverse_kinematics_solution_success_flag:
                target_joint_position_r = self.right_arm_interpolation_result
                target_joint_position =np.array(current_joint_position[:7].tolist() + target_joint_position_r.tolist() + current_joint_position[14:].tolist())
                # end_posi = np.array(target_joint_position[:7].tolist() + [3.14/2.5, -0.4, -3.14/1.75,  3.14/1.5,  3.14/2, 0, 0] + target_joint_position[14:].tolist())
                end_posi = np.array(current_joint_position[:7].tolist() + [-0.3, -0.2, -0.5, 0.6, -0.5, 0, 0] + current_joint_position[14:].tolist())
            else:
                print("❌ 右臂逆解失败")
                return False
        else:
            self.left_arm_inverse_kinematics(R, xyz, np.array(current_joint_position[:7]))
            if self.left_arm_inverse_kinematics_solution_success_flag:
                target_joint_position_l = self.left_arm_interpolation_result
                target_joint_position =np.array(target_joint_position_l.tolist() + current_joint_position[7:14].tolist() + current_joint_position[14:].tolist())
                # end_posi = np.array([3.14/2.5,  0.4,  3.14/1.75, -3.14/1.5, -3.14/2, 0, 0] + target_joint_position[7:14].tolist() + target_joint_position[14:].tolist())
                end_posi = np.array([-0.1,  0.2,  0.5, 0.3,  0.5, 0, 0] + current_joint_position[7:].tolist())
            else:
                print("❌ 左臂逆解失败")
                return False
        
        self.MOVEL.moveL2targetjointposition(current_joint_position, target_joint_position)
        self.MOVEL.moveL2targetjointposition(target_joint_position, end_posi)
        # self.MOVEJ.moveJ2target(target_joint_position, end_posi)
        return True

    def move_relative_FT(self, arm, delta_xyz, delta_rpy):
        """
        相对当前位置移动 delta_xyz（保持姿态）
        """
        if self.MOVEL is None:
            print("❌ MOVEL 未设置")
            return False

        current_joint_position = self.lcm_handler.joint_current_pos.copy()

        # 当前末端位姿
        if arm == 'right':
            current_pose = self.right_arm_forward_kinematics(current_joint_position[7:14])
        else:
            current_pose = self.left_arm_forward_kinematics(current_joint_position[:7])

        current_position = current_pose.translation
        target_position = current_position + delta_xyz
        R = pinocchio.rpy.rpyToMatrix(delta_rpy) @ current_pose.rotation  # 保持姿态

        # 求逆解
        if arm == 'right':
            self.right_arm_inverse_kinematics(R, target_position, current_joint_position[7:14])
            if self.right_arm_inverse_kinematics_solution_success_flag:
                target_joint_position_r = self.right_arm_interpolation_result
                target_joint_position =np.array(current_joint_position[:7].tolist() + target_joint_position_r.tolist() + current_joint_position[14:].tolist())
            else:
                print("❌ 右臂逆解失败")
                return False
        else:
            self.left_arm_inverse_kinematics(R, target_position, current_joint_position[:7])
            if self.left_arm_inverse_kinematics_solution_success_flag:
                target_joint_position_l = self.left_arm_interpolation_result
                target_joint_position =np.array(target_joint_position_l.tolist() + current_joint_position[7:].tolist())
            else:
                print("❌ 左臂逆解失败")
                return False

        # 执行相对运动
        self.MOVEL.moveL2targetjointposition_FT(current_joint_position, target_joint_position)
        return True
