import os
import pinocchio
import numpy as np
from numpy.linalg import norm, pinv
import time 
import copy


class Kinematic_Model:
    def __init__(self):

        parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(parent_folder, "models", "P5_left_arm.urdf")
        self.left_arm_pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.left_arm_pin_data = self.left_arm_pin_model.createData()
        print('model name: ' + self.left_arm_pin_model.name)


        urdf_path = os.path.join(parent_folder, "models", "P5_right_arm.urdf")
        self.right_arm_pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.right_arm_pin_data = self.right_arm_pin_model.createData()
        print('model name: ' + self.right_arm_pin_model.name)

        self.left_arm_interpolation_result = np.zeros(7)
        self.right_arm_interpolation_result = np.zeros(7)

        self.left_arm_inverse_kinematics_solution_success_flag = None
        self.right_arm_inverse_kinematics_solution_success_flag = None

        

    def left_arm_forward_kinematics(self, left_arm_joint_position):
        pinocchio.forwardKinematics(self.left_arm_pin_model, self.left_arm_pin_data, left_arm_joint_position)
        left_arm_cart_pose = copy.deepcopy(self.left_arm_pin_data.oMi[self.left_arm_pin_model.njoints - 1])
        return left_arm_cart_pose
    

    def right_arm_forward_kinematics(self, right_arm_joint_position):
        pinocchio.forwardKinematics(self.right_arm_pin_model, self.right_arm_pin_data, right_arm_joint_position)
        right_arm_cart_pose = copy.deepcopy(self.right_arm_pin_data.oMi[self.right_arm_pin_model.njoints - 1])
        return right_arm_cart_pose
    

    def left_arm_inverse_kinematics(self, cart_interpolation_pose, cart_interpolation_position, current_joint_position):
        oMdes = pinocchio.SE3(np.array(cart_interpolation_pose), np.array(cart_interpolation_position))
        eps    = 1e-7
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-12
        q = current_joint_position
        i = 1 
        while(1):
            pinocchio.forwardKinematics(self.left_arm_pin_model, self.left_arm_pin_data, q)
            pinocchio.updateFramePlacements(self.left_arm_pin_model, self.left_arm_pin_data) 
            iMd = self.left_arm_pin_data.oMi[7].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame
            if np.linalg.norm(err) < eps:
                self.left_arm_interpolation_result = q
                self.left_arm_inverse_kinematics_solution_success_flag = True
                # print("逆解循环了:{} 次".format(i))
                break
            if i >= IT_MAX:
                self.left_arm_inverse_kinematics_solution_success_flag = False
                self.left_arm_interpolation_result = current_joint_position
                break

            pinocchio.computeJointJacobians(self.left_arm_pin_model, self.left_arm_pin_data)
            J = pinocchio.getJointJacobian(self.left_arm_pin_model, self.left_arm_pin_data, self.left_arm_pin_model.njoints - 1, pinocchio.LOCAL)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.left_arm_pin_model, q, v * DT)

            i += 1

    def right_arm_inverse_kinematics(self, cart_interpolation_pose, cart_interpolation_position, current_joint_position):
        oMdes = pinocchio.SE3(np.array(cart_interpolation_pose), np.array(cart_interpolation_position))
        eps    = 1e-7
        IT_MAX = 1000
        DT     = 1e-1
        damp   = 1e-12
        q = current_joint_position
        i = 1 
        while(1):
            pinocchio.forwardKinematics(self.right_arm_pin_model, self.right_arm_pin_data, q)
            pinocchio.updateFramePlacements(self.right_arm_pin_model, self.right_arm_pin_data) 
            iMd = self.right_arm_pin_data.oMi[7].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame
            if np.linalg.norm(err) < eps:
                self.right_arm_interpolation_result = q
                self.right_arm_inverse_kinematics_solution_success_flag = True
                # print("逆解循环了:{} 次".format(i))
                break
            if i >= IT_MAX:
                self.right_arm_inverse_kinematics_solution_success_flag = False
                self.right_arm_interpolation_result = current_joint_position
                break

            pinocchio.computeJointJacobians(self.right_arm_pin_model, self.right_arm_pin_data)
            J = pinocchio.getJointJacobian(self.right_arm_pin_model, self.right_arm_pin_data, self.right_arm_pin_model.njoints - 1, pinocchio.LOCAL)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.right_arm_pin_model, q, v * DT)

            i += 1

        



