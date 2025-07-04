import numpy as np
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

# 求逆解数值解
def arm_inverse_kinematics(target_pose, para, pos_tcp,joint_limit, theta_current,theta_analysis):
    """
    参数:
    target_pose: 目标位姿，形状为(4, 4)
    para: 机械臂运动学参数＋末端位置，形状为 (45,)
    joint_limit: 机械臂关节限位，形状为(2,7)
    theta_current (numpy.ndarray): 当前关节角，形状为 (7,)
    返回:
    theta_best (numpy.ndarray): 最优关节角解
    error_info (str): 错误信息
    """
    target_pose_armbase = np.linalg.inv(rpy_pos2h(para[0:6])) @ target_pose
    para_armbase = para.copy()
    para_armbase[0:6] = np.zeros(6, )
    # 求解析解初值
    theta_init, error_info = arm_inverse_kinematics_analytical(target_pose, para, pos_tcp, joint_limit, theta_current,theta_analysis)
    theta_analysis = theta_init.copy()
    _, pose_init = arm_forward_kinematics(theta_init, para, pos_tcp)
    error_pos_ik = target_pose[:3, 3] - pose_init[:3, 3]

    # 雅可比迭代
    while np.linalg.norm(error_pos_ik) > 1e-10:
        _, pose_init = arm_forward_kinematics(theta_init, para, pos_tcp)
        Jv = get_jacobian_velocity(para, theta_init, pos_tcp)
        error_pos_ik = target_pose[:3, 3] - pose_init[:3, 3]
        error_div_ik = rotation_matrix_to_axis_angle(target_pose[:3, :3] @ pose_init[:3, :3].T)
        error_ik = np.hstack((error_pos_ik, error_div_ik))
        dag = pinv(Jv) @ error_ik
        theta_init = theta_init + dag
    theta_best = theta_init
    return theta_best, theta_analysis, error_info

# 求逆解解析解
def arm_inverse_kinematics_analytical(target_pose, para, pos_tcp, joint_limit, theta_current,theta_analysis):
    """
    参数:
    target_pose (numpy.ndarray): 目标位姿，形状为 (4, 4)
    para (numpy.ndarray): 机械臂参数
    joint_limit (numpy.ndarray): 关节限制，形状为 (2, 7)
    theta_current (numpy.ndarray): 当前关节角，形状为 (7,)
    返回:
    theta_best (numpy.ndarray): 最优关节角解
    error_info (str): 错误信息
    """
    # 初始化
    theta_best = np.zeros(7)

    # 计算连杆长度
    mSE = rpy_pos2h(para[6:12]) @ rpy_pos2h(para[12:18]) @ rpy_pos2h(para[18:24])
    dSE = np.linalg.norm(mSE[:3, 3])
    mEW = rpy_pos2h(para[24:30]) @ rpy_pos2h(para[30:36]) @ rpy_pos2h(para[36:42])
    dEW = np.linalg.norm(mEW[:3, 3])
    m_tool = np.eye(4)
    m_tool[:3, 3] = pos_tcp
    mSW = target_pose @ np.linalg.inv(m_tool)
    vecSW = mSW[:3, 3]
    dSW = np.linalg.norm(vecSW)

    # 工作空间判断
    if (dSE + dEW) <= dSW:
        theta_best = theta_current
        error_info = 'Unreachable'
        print("Warning: 该位姿超出工作空间")
        return theta_best, error_info

    # 求解关节4
    theta4 = np.array([
        np.pi - np.arccos((dSE ** 2 + dEW ** 2 - dSW ** 2) / (2 * dSE * dEW)),
        np.arccos((dSE ** 2 + dEW ** 2 - dSW ** 2) / (2 * dSE * dEW)) - np.pi
    ])

    # 构造肘部姿态
    alpha = np.arccos((dSE ** 2 + dSW ** 2 - dEW ** 2) / (2 * dSE * dSW))
    vecSV = np.array([0, 1, 0])
    vecL = np.cross(vecSW, vecSV)
    vecL = vecL / np.linalg.norm(vecL)
    mLx = get_object_m(vecL)
    R_l_alpha = np.zeros((3, 3, 2))
    R_l_alpha[:, :, 0] = np.eye(3) + np.sin(-alpha) * mLx + (1 - np.cos(-alpha)) * mLx @ mLx
    R_l_alpha[:, :, 1] = np.eye(3) + np.sin(alpha) * mLx + (1 - np.cos(alpha)) * mLx @ mLx
    SE_0 = np.zeros((3, 2))
    SE_0[:, 0] = dSE * R_l_alpha[:, :, 0] @ (vecSW / dSW)
    SE_0[:, 1] = dSE * R_l_alpha[:, :, 1] @ (vecSW / dSW)
    z03_0 = np.zeros((3, 2))
    z03_0[:, 0] = -SE_0[:, 0] / np.linalg.norm(SE_0[:, 0])
    z03_0[:, 1] = -SE_0[:, 1] / np.linalg.norm(SE_0[:, 1])
    x03_0 = np.zeros((3, 2))
    x03_0[:, 0] = vecL
    x03_0[:, 1] = -vecL
    y03_0 = np.zeros((3, 4))
    y03_0[:, 0] = np.cross(z03_0[:, 0], x03_0[:, 0])
    y03_0[:, 1] = np.cross(z03_0[:, 0], x03_0[:, 1])
    y03_0[:, 2] = np.cross(z03_0[:, 1], x03_0[:, 0])
    y03_0[:, 3] = np.cross(z03_0[:, 1], x03_0[:, 1])
    R03_0 = np.zeros((3, 3, 4))
    R03_0[:, :, 0] = np.column_stack((x03_0[:, 0], y03_0[:, 0], z03_0[:, 0]))
    R03_0[:, :, 1] = np.column_stack((x03_0[:, 1], y03_0[:, 1], z03_0[:, 0]))
    R03_0[:, :, 2] = np.column_stack((x03_0[:, 0], y03_0[:, 2], z03_0[:, 1]))
    R03_0[:, :, 3] = np.column_stack((x03_0[:, 1], y03_0[:, 3], z03_0[:, 1]))
    vecSW = vecSW / dSW
    mSWx = get_object_m(vecSW)

    # 求解臂型角有效区间
    print('case 1')
    psi_interval1 = get_psi_interval(mSW, mSWx, R03_0[:, :, 0], theta4[0], joint_limit)
    print('case 2')
    psi_interval2 = get_psi_interval(mSW, mSWx, R03_0[:, :, 1], theta4[1], joint_limit)
    print('case 3')
    psi_interval3 = get_psi_interval(mSW, mSWx, R03_0[:, :, 2], theta4[1], joint_limit)
    print('case 4')
    psi_interval4 = get_psi_interval(mSW, mSWx, R03_0[:, :, 3], theta4[0], joint_limit)
    psi_interval = merge_intervals(np.hstack((psi_interval1, psi_interval2, psi_interval3, psi_interval4)))

    if psi_interval.size == 0:
        error_info = 'noSolve'
        return theta_best, error_info

    # 求解当前关节角对应的臂角
    psi_current = get_psi_by_joint_angles(theta_analysis, para, pos_tcp)
    psi_best = psi_opt(psi_current, theta_analysis, target_pose, joint_limit, para, psi_interval, pos_tcp)
    theta_best, error_info = srs_inverse_kinematics(target_pose, para, joint_limit, psi_best, theta_current, pos_tcp)

    return theta_best, error_info

# 计算最佳臂角
def psi_opt(psi_current, theta_current, target_pose, joint_limit, para, psi_interval, pos_tcp):
    """
    参数:
    psi_current (float): 当前臂角
    theta_current (numpy.ndarray): 当前关节角，形状为 (7,)
    target_pose (numpy.ndarray): 目标位姿，形状为 (4, 4)
    joint_limit (numpy.ndarray): 关节限制，形状为 (2, 7)
    para (numpy.ndarray): 机械臂参数
    psi_interval (numpy.ndarray): 臂角区间，形状为 (2, n)
    返回:
    psi_best (float): 最佳臂角
    """
    # 计算关节归一化处理结果
    xt = np.zeros(7)
    for i in range(7):
        xt[i] = (2 / (joint_limit[1, i] - joint_limit[0, i])) * (
                    theta_current[i] - (joint_limit[1, i] + joint_limit[0, i]) / 2)

    a = 2
    b = 2
    wx = np.zeros(7)
    for i in range(7):
        if xt[i] < 0:
            wx[i] = (-b * xt[i]) / (np.exp((1 + xt[i]) * a) - 1)
        else:
            wx[i] = (b * xt[i]) / (np.exp((1 - xt[i]) * a) - 1)

    delta_psi = 0.001
    theta_best_0, _ = srs_inverse_kinematics(target_pose, para, joint_limit, psi_current, theta_current, pos_tcp)
    theta_best_1, _ = srs_inverse_kinematics(target_pose, para, joint_limit, psi_current + delta_psi, theta_current, pos_tcp)
    alpha = (theta_best_1 - theta_best_0) / delta_psi

    M = np.sum(wx * (alpha ** 2))
    N = 2 * np.sum(wx * alpha * (theta_best_0 - theta_current - alpha * psi_current))
    K = np.sum(wx * ((theta_best_0 - theta_current - alpha * psi_current) ** 2))

    best_psi_alter = -N / (2 * M)

    # 检查最佳臂角是否在可行区间内
    for i in range(psi_interval.shape[1]):
        if psi_interval[0, i] <= best_psi_alter <= psi_interval[1, i]:
            return best_psi_alter

    # 寻找最近的有效臂角
    to_edge = np.min(np.abs(best_psi_alter - psi_interval), axis=0)
    index = np.argmin(to_edge)
    dist_left = abs(best_psi_alter - psi_interval[0, index])
    dist_right = abs(best_psi_alter - psi_interval[1, index])

    if dist_left < dist_right:
        return psi_interval[0, index]
    else:
        return psi_interval[1, index]

# 已知关节角的情况下求解臂角
def get_psi_by_joint_angles(angle, para, pos_tcp):
    """
    参数:
    angle (numpy.ndarray): 关节角，形状为 (7,)
    para (numpy.ndarray): 机械臂参数
    返回:
    psi (float): 臂角
    """
    # 计算连杆长度
    mSE = rpy_pos2h(para[6:12]) @ rpy_pos2h(para[12:18]) @ rpy_pos2h(para[18:24])
    dSE = np.linalg.norm(mSE[:3, 3])
    mEW = rpy_pos2h(para[24:30]) @ rpy_pos2h(para[30:36]) @ rpy_pos2h(para[36:42])
    dEW = np.linalg.norm(mEW[:3, 3])

    # 正运动学
    t_jointi, m_tcp = arm_forward_kinematics(angle, para, pos_tcp)
    m_tool = np.eye(4)
    m_tool[:3, 3] = pos_tcp
    mSW = m_tcp @ np.linalg.inv(m_tool)
    vecSW = mSW[:3, 3]
    vecSE = t_jointi[:3, 3, 3]
    dSW = np.linalg.norm(vecSW)

    # 计算theta_ESW
    theta_ESW = optimized_acos((dSE ** 2 + dSW ** 2 - dEW ** 2) / (2 * dSE * dSW))

    # 计算单位向量uSW
    uSW = vecSW / dSW
    vecSV = np.array([0, 1, 0])

    # 计算向量k
    k = np.cross(uSW, vecSV) / np.linalg.norm(np.cross(uSW, vecSV))

    # 计算反对称矩阵K
    K = get_object_m(k)

    # 计算R_k_theta_ESW
    R_k_theta_ESW = np.eye(3) + np.sin(theta_ESW) * K + (1 - np.cos(theta_ESW)) * K @ K

    # 计算vecSE0
    vecSE0 = R_k_theta_ESW @ uSW * dSE

    # 计算foot
    foot = np.dot(vecSE, vecSW) / dSW * uSW

    # 计算pFE0和pFE
    pFE0 = vecSE0 - foot
    pFE = t_jointi[:3, 3, 3] - foot

    # 计算psi
    psi = np.sign(np.dot(uSW, np.cross(pFE0, pFE))) * optimized_acos(
        np.dot(pFE0, pFE) / (np.linalg.norm(pFE0) * np.linalg.norm(pFE)))

    return psi

# 获取臂角有效区间
def get_psi_interval(mSW, mSWx, R03_0, theta4, joint_limit):
    """
    参数:
    mSW (numpy.ndarray): 目标位姿的旋转部分
    mSWx (numpy.ndarray): 目标位姿的反对称矩阵
    R03_0 (numpy.ndarray): 关节3的旋转矩阵
    theta4 (float): 关节4的角度
    joint_limit (numpy.ndarray): 关节限制，形状为 (2, 7)

    返回:
    psi_interval (numpy.ndarray): 臂角的有效区间
    """
    # 123关节abc
    aS = mSWx @ R03_0
    bS = -mSWx @ mSWx @ R03_0
    cS = (np.eye(3) + mSWx @ mSWx) @ R03_0

    # 567关节abc
    R34 = rot_about_axis(theta4, 'x')
    R43 = R34.T
    R07 = mSW[:3, :3]
    aW = R43 @ aS.T @ R07
    bW = R43 @ bS.T @ R07
    cW = R43 @ cS.T @ R07

    # 判断关节4是否在限位中
    if theta4 > np.pi:
        theta4 -= 2 * np.pi
    if theta4 < -np.pi:
        theta4 += 2 * np.pi
    if theta4 <= joint_limit[0, 3] or theta4 >= joint_limit[1, 3]:
        print("Warning: 该位姿下无有效臂角，即无关节逆解")
        return np.array([]).reshape(2, 0)

    # sin型关节角有效臂角范围
    feapsi_2 = get_sin_psi_interval(joint_limit, aS, bS, cS, 2)
    feapsi_6 = get_sin_psi_interval(joint_limit, aW, bW, cW, 6)

    # tan型关节角有效臂角范围
    feapsi_1 = get_tan_psi_interval(joint_limit, aS, bS, cS, 1)
    feapsi_3 = get_tan_psi_interval(joint_limit, aS, bS, cS, 3)
    feapsi_5 = get_tan_psi_interval(joint_limit, aW, bW, cW, 5)
    feapsi_7 = get_tan_psi_interval(joint_limit, aW, bW, cW, 7)

    # 求解关节对应有效臂角范围交集
    feapsi_12 = com_multi_interval(feapsi_1, feapsi_2)
    feapsi_123 = com_multi_interval(feapsi_12, feapsi_3)
    feapsi_12345 = com_multi_interval(feapsi_123, feapsi_5)
    feapsi_123456 = com_multi_interval(feapsi_12345, feapsi_6)
    feapsi_1234567 = com_multi_interval(feapsi_123456, feapsi_7)

    psi_interval = feapsi_1234567
    if psi_interval.size == 0:
        return np.array([]).reshape(2, 0)
    else:
        return psi_interval

# 求解 Sin 臂角范围
def get_sin_psi_interval(joint_limit, a, b, c, joint_no):
    """
    参数:
    joint_limit (numpy.ndarray): 关节限制，形状为 (2, 7)，第一行为下限，第二行为上限
    a (numpy.ndarray): a 矩阵
    b (numpy.ndarray): b 矩阵
    c (numpy.ndarray): c 矩阵
    joint_no (int): 关节编号，2 或 6
    返回:
    psi_interval (numpy.ndarray): 臂角范围，形状为 (2,)
    """
    if joint_no == 2:
        aJ = -a[1, 2]
        bJ = -b[1, 2]
        cJ = -c[1, 2]
        qL = joint_limit[0, 1]
        qU = joint_limit[1, 1]
    elif joint_no == 6:
        aJ = a[2, 1]
        bJ = b[2, 1]
        cJ = c[2, 1]
        qL = joint_limit[0, 5]
        qU = joint_limit[1, 5]
    else:
        raise ValueError("关节编号必须为 2 或 6")

    fai = np.arctan2(bJ, aJ)

    if aJ == 0 and bJ == 0:
        psi_interval = np.array([-np.pi, np.pi])
    else:
        theta_interval = np.linspace(qL, qU, 1000)
        cMin = (np.min(np.sin(theta_interval)) - cJ) / np.sqrt(aJ ** 2 + bJ ** 2)
        cMax = (np.max(np.sin(theta_interval)) - cJ) / np.sqrt(aJ ** 2 + bJ ** 2)
        psi_interval = sin_angle_judge(cMin, cMax, fai)

    return psi_interval

# 根据 cMin 和 cMax 的值判断臂角范围
def sin_angle_judge(cMin, cMax, fai):
    """
    参数:
    cMin (float): 最小值
    cMax (float): 最大值
    fai (float): 角度
    返回:
    psi_interval (numpy.ndarray): 臂角范围，形状为 (2, n)
    """
    if cMin >= 1 or cMax <= -1:
        psi_interval = np.array([])
    elif cMin < 1 and cMin >= 0 and cMax >= 1:
        psi_interval = np.array([
            [np.arcsin(cMin) - 2 * np.pi - fai, -np.pi - np.arcsin(cMin) - fai],
            [np.arcsin(cMin) - fai, np.pi - np.arcsin(cMin) - fai]
        ])
    elif cMin < 1 and cMin >= 0 and cMax < 1 and cMax > 0:
        psi_interval = np.array([
            [np.arcsin(cMin) - 2 * np.pi - fai, np.arcsin(cMax) - 2 * np.pi - fai],
            [-np.pi - np.arcsin(cMax) - fai, -np.pi - np.arcsin(cMin) - fai],
            [np.arcsin(cMin) - fai, np.arcsin(cMax) - fai],
            [np.pi - np.arcsin(cMax) - fai, np.pi - np.arcsin(cMin) - fai]
        ])
    elif cMin < 1 and cMin >= 0 and cMax <= 0:
        psi_interval = np.array([])
    elif cMin < 0 and cMin >= -1 and cMax >= 1:
        psi_interval = np.array([
            [-2 * np.pi - fai, -np.pi - np.arcsin(cMin) - fai],
            [np.arcsin(cMin) - fai, np.pi - np.arcsin(cMin) - fai],
            [2 * np.pi + np.arcsin(cMin) - fai, 2 * np.pi - fai]
        ])
    elif cMin < 0 and cMin >= -1 and cMax < 1 and cMax >= 0:
        psi_interval = np.array([
            [-2 * np.pi - fai, np.arcsin(cMax) - 2 * np.pi - fai],
            [-np.pi - np.arcsin(cMax) - fai, -np.pi - np.arcsin(cMin) - fai],
            [np.arcsin(cMin) - fai, np.arcsin(cMax) - fai],
            [np.pi - np.arcsin(cMax) - fai, np.pi - np.arcsin(cMin) - fai],
            [2 * np.pi + np.arcsin(cMin) - fai, 2 * np.pi + np.arcsin(cMax) - fai]
        ])
    elif cMin < 0 and cMin >= -1 and cMax < 0 and cMax >= -1:
        psi_interval = np.array([
            [-np.pi - np.arcsin(cMax) - fai, -np.pi - np.arcsin(cMin) - fai],
            [np.arcsin(cMin) - fai, np.arcsin(cMax) - fai],
            [np.pi - np.arcsin(cMax) - fai, np.pi - np.arcsin(cMin) - fai],
            [2 * np.pi + np.arcsin(cMin) - fai, 2 * np.pi + np.arcsin(cMax) - fai]
        ])
    elif cMin < 0 and cMin >= -1 and cMax < -1:
        psi_interval = np.array([])
    elif cMin < -1 and cMax >= 1:
        psi_interval = np.array([[-np.pi, np.pi]])
    elif cMin < -1 and cMax < 1 and cMax >= 0:
        psi_interval = np.array([
            [-2 * np.pi - fai, np.arcsin(cMax) - 2 * np.pi - fai],
            [-np.pi - np.arcsin(cMax) - fai, np.arcsin(cMax) - fai],
            [np.pi - np.arcsin(cMax) - fai, 2 * np.pi - fai]
        ])
    elif cMin < -1 and cMax < 0 and cMax > -1:
        psi_interval = np.array([
            [-np.pi - np.arcsin(cMax) - fai, np.arcsin(cMax) - fai],
            [np.pi - np.arcsin(cMax) - fai, 2 * np.pi + np.arcsin(cMax) - fai]
        ])

    # 越界处理
    if psi_interval.size > 0:
        psi_interval = psi_interval.T
        no1 = np.where(psi_interval[0, :] > np.pi)[0]
        no2 = np.where(psi_interval[1, :] < -np.pi)[0]
        no = np.concatenate((no1, no2))
        psi_interval = np.delete(psi_interval, no, axis=1)
        psi_interval[psi_interval > np.pi] = np.pi
        psi_interval[psi_interval < -np.pi] = -np.pi

    return psi_interval

# 求解 tan 臂角范围
def get_tan_psi_interval(joint_limit, a, b, c, joint_no):
    """
    参数:
    joint_limit (numpy.ndarray): 关节限制，形状为 (2, 7)，第一行为下限，第二行为上限
    a (numpy.ndarray): a 矩阵
    b (numpy.ndarray): b 矩阵
    c (numpy.ndarray): c 矩阵
    joint_no (int): 关节编号，1,3,5,7
    返回:
    psi_interval (numpy.ndarray): 臂角范围，形状为 (2,)
    """
    if joint_no == 1:
        qL = joint_limit[0, 0]
        qU = joint_limit[1, 0]
        an = a[0, 2]
        bn = b[0, 2]
        cn = c[0, 2]
        ad = a[2, 2]
        bd = b[2, 2]
        cd = c[2, 2]
    elif joint_no == 3:
        qL = joint_limit[0, 2]
        qU = joint_limit[1, 2]
        an = a[1, 0]
        bn = b[1, 0]
        cn = c[1, 0]
        ad = a[1, 1]
        bd = b[1, 1]
        cd = c[1, 1]
    elif joint_no == 5:
        qL = joint_limit[0, 4]
        qU = joint_limit[1, 4]
        an = -a[0, 1]
        bn = -b[0, 1]
        cn = -c[0, 1]
        ad = a[1, 1]
        bd = b[1, 1]
        cd = c[1, 1]
    elif joint_no == 7:
        qL = joint_limit[0, 6]
        qU = joint_limit[1, 6]
        an = -a[2, 0]
        bn = -b[2, 0]
        cn = -c[2, 0]
        ad = a[2, 2]
        bd = b[2, 2]
        cd = c[2, 2]
    else:
        raise ValueError("关节编号必须为 1, 3, 5 或 7")

    at = bd * cn - bn * cd
    bt = an * cd - ad * cn
    ct = an * bd - ad * bn
    q_edge = np.arctan2(cn - bn, cd - bd)

    a2_u = -cn + np.tan(qU) * cd + bn - np.tan(qU) * bd
    b_u = -an + np.tan(qU) * ad
    bb_4ac_u = (an - np.tan(qU) * ad) ** 2 + (np.tan(qU) * bd - bn) ** 2 - (cn - np.tan(qU) * cd) ** 2

    a2_l = -cn + np.tan(qL) * cd + bn - np.tan(qL) * bd
    b_l = -an + np.tan(qL) * ad
    bb_4ac_l = (an - np.tan(qL) * ad) ** 2 + (np.tan(qL) * bd - bn) ** 2 - (cn - np.tan(qL) * cd) ** 2

    psi1_qu = 2 * np.arctan((-b_u - np.sqrt(bb_4ac_u)) / (a2_u))
    sinq_psi1_qu = an * np.sin(psi1_qu) + bn * np.cos(psi1_qu) + cn
    cosq_psi1_qu = ad * np.sin(psi1_qu) + bd * np.cos(psi1_qu) + cd

    psi2_qu = 2 * np.arctan((-b_u + np.sqrt(bb_4ac_u)) / (a2_u))

    psi1_ql = 2 * np.arctan((-b_l - np.sqrt(bb_4ac_l)) / (a2_l))
    sinq_psi1_ql = an * np.sin(psi1_ql) + bn * np.cos(psi1_ql) + cn
    cosq_psi1_ql = ad * np.sin(psi1_ql) + bd * np.cos(psi1_ql) + cd

    psi2_ql = 2 * np.arctan((-b_l + np.sqrt(bb_4ac_l)) / (a2_l))

    if at ** 2 + bt ** 2 - ct ** 2 > 0:
        psi_min = 2 * np.arctan((at - np.sqrt(at ** 2 + bt ** 2 - ct ** 2)) / (bt - ct))
        psi_max = 2 * np.arctan((at + np.sqrt(at ** 2 + bt ** 2 - ct ** 2)) / (bt - ct))
        psi_left = min(psi_min, psi_max)
        sin_psi_left = an * np.sin(psi_left) + bn * np.cos(psi_left) + cn
        cos_psi_left = ad * np.sin(psi_left) + bd * np.cos(psi_left) + cd
        q_left = np.arctan2(sin_psi_left, cos_psi_left)
        psi_right = max(psi_min, psi_max)
        sin_psi_right = an * np.sin(psi_right) + bn * np.cos(psi_right) + cn
        cos_psi_right = ad * np.sin(psi_right) + bd * np.cos(psi_right) + cd
        q_right = np.arctan2(sin_psi_right, cos_psi_right)

        q_min = min(q_left, q_right)
        q_max = max(q_left, q_right)

    if at ** 2 + bt ** 2 - ct ** 2 < 0:  # 单调型
        if (sinq_psi1_qu > 0) and (cosq_psi1_qu < 0):
            psi_qu = psi1_qu
        else:
            psi_qu = psi2_qu
        if (sinq_psi1_ql < 0) and (cosq_psi1_ql < 0):
            psi_ql = psi1_ql
        else:
            psi_ql = psi2_ql
        if ct > 0:
            if psi_ql <= psi_qu:
                psi_interval = np.array([[psi_ql], [psi_qu]])
            else:
                psi_interval = np.array([[-np.pi, psi_ql], [psi_qu, np.pi]])
        else:
            if psi_ql >= psi_qu:
                psi_interval = np.array([[psi_qu], [psi_ql]])
            else:
                psi_interval = np.array([[-np.pi, psi_qu], [psi_ql, np.pi]])
    elif at ** 2 + bt ** 2 - ct ** 2 > 0:  # 周期型
        if (q_min > qU) or (q_max < qL):
            psi_interval = np.array([])
        elif (q_min >= qL) and (q_min <= qU) and (q_max >= qL) and (q_max <= qU):
            psi_interval = np.array([[-np.pi], [np.pi]])
        elif (q_min < qL) and (q_max >= qL) and (q_max <= qU):
            if qL > q_edge:
                psi_interval = np.array([[min(psi1_ql, psi2_ql)], [max(psi1_ql, psi2_ql)]])
            else:
                psi_interval = np.array([[-np.pi, max(psi1_ql, psi2_ql)], [min(psi1_ql, psi2_ql), np.pi]])
        elif (q_min >= qL) and (q_min <= qU) and (q_max > qU):
            if qU <= q_edge:
                psi_interval = np.array([[min(psi1_qu, psi2_qu)], [max(psi1_qu, psi2_qu)]])
            else:
                psi_interval = np.array([[-np.pi, max(psi1_qu, psi2_qu)], [min(psi1_qu, psi2_qu), np.pi]])
        else:
            if (qL <= q_edge) and (qU <= q_edge):
                psi_interval = np.array(
                    [[min(psi1_qu, psi2_qu), max(psi1_ql, psi2_ql)], [min(psi1_ql, psi2_ql), max(psi1_qu, psi2_qu)]])
            elif (qL <= q_edge) and (qU > q_edge):
                if q_left >= q_right:
                    psi_interval = np.array([[-np.pi, max(psi1_qu, psi2_qu), max(psi1_ql, psi2_ql)],
                                             [min(psi1_qu, psi2_qu), min(psi1_ql, psi2_ql), np.pi]])
                else:
                    psi_interval = np.array([[-np.pi, max(psi1_ql, psi2_ql), max(psi1_qu, psi2_qu)],
                                             [min(psi1_ql, psi2_ql), min(psi1_qu, psi2_qu), np.pi]])
            else:
                psi_interval = np.array(
                    [[min(psi1_ql, psi2_ql), max(psi1_qu, psi2_qu)], [min(psi1_qu, psi2_qu), max(psi1_ql, psi2_ql)]])
    else:  # 算法奇异
        print("Warning: 算法奇异")
        psi_interval = np.array([])

    return psi_interval

# SRS构型机械臂解析解求解。
def srs_inverse_kinematics(target_pose, para, joint_limit, psi, theta_current, pos_tcp):
    """
    参数:
    target_pose (numpy.ndarray): 目标位姿，形状为 (4, 4) 的齐次变换矩阵
    para (numpy.ndarray): 机械臂参数
    joint_limit (numpy.ndarray): 关节限制，形状为 (2, 7)，第一行为下限，第二行为上限
    psi (float): 臂角
    theta_current (numpy.ndarray): 当前关节角，形状为 (7,)
    返回:
    theta_best (numpy.ndarray): 最优关节角解
    error_info (str): 错误信息
    """
    # 初始化
    error_info = 'Reachable'
    # 计算连杆长度
    mSE = rpy_pos2h(para[6:12]) @ rpy_pos2h(para[12:18]) @ rpy_pos2h(para[18:24])
    dSE = np.linalg.norm(mSE[:3, 3])
    mEW = rpy_pos2h(para[24:30]) @ rpy_pos2h(para[30:36]) @ rpy_pos2h(para[36:42])
    dEW = np.linalg.norm(mEW[:3, 3])
    m_tool = np.eye(4)
    m_tool[:3, 3] = pos_tcp
    mSW = target_pose @ np.linalg.inv(m_tool)
    vecSW = mSW[:3, 3]
    dSW = np.linalg.norm(vecSW)

    # 工作空间可达判断 & 求解关节4
    if (dSE + dEW) < dSW:
        theta_best = theta_current
        error_info = 'Unreachable'
        return theta_best, error_info
    else:
        theta4 = np.array([
            np.pi - np.arccos((dSE ** 2 + dEW ** 2 - dSW ** 2) / (2 * dSE * dEW)),
            np.arccos((dSE ** 2 + dEW ** 2 - dSW ** 2) / (2 * dSE * dEW)) - np.pi
        ])

    # 构造肘部姿态
    alpha = np.arccos((dSE ** 2 + dSW ** 2 - dEW ** 2) / (2 * dSE * dSW))
    vecSV = np.array([0, 1, 0])
    vecL = np.cross(vecSW, vecSV)
    vecL = vecL / np.linalg.norm(vecL)
    mLx = get_object_m(vecL)
    R_l_alpha = np.zeros((3, 3, 2))
    R_l_alpha[:, :, 0] = np.eye(3) + np.sin(-alpha) * mLx + (1 - np.cos(-alpha)) * mLx @ mLx
    R_l_alpha[:, :, 1] = np.eye(3) + np.sin(alpha) * mLx + (1 - np.cos(alpha)) * mLx @ mLx
    SE_0 = np.zeros((3, 2))
    SE_0[:, 0] = dSE * R_l_alpha[:, :, 0] @ (vecSW / dSW)
    SE_0[:, 1] = dSE * R_l_alpha[:, :, 1] @ (vecSW / dSW)
    z03_0 = np.zeros((3, 2))
    z03_0[:, 0] = -SE_0[:, 0] / np.linalg.norm(SE_0[:, 0])
    z03_0[:, 1] = -SE_0[:, 1] / np.linalg.norm(SE_0[:, 1])
    x03_0 = np.zeros((3, 2))
    x03_0[:, 0] = vecL
    x03_0[:, 1] = -vecL
    y03_0 = np.zeros((3, 4))
    y03_0[:, 0] = np.cross(z03_0[:, 0], x03_0[:, 0])
    y03_0[:, 1] = np.cross(z03_0[:, 0], x03_0[:, 1])
    y03_0[:, 2] = np.cross(z03_0[:, 1], x03_0[:, 0])
    y03_0[:, 3] = np.cross(z03_0[:, 1], x03_0[:, 1])
    R03_0 = np.zeros((3, 3, 4))
    R03_0[:, :, 0] = np.column_stack((x03_0[:, 0], y03_0[:, 0], z03_0[:, 0]))
    R03_0[:, :, 1] = np.column_stack((x03_0[:, 1], y03_0[:, 1], z03_0[:, 0]))
    R03_0[:, :, 2] = np.column_stack((x03_0[:, 0], y03_0[:, 2], z03_0[:, 1]))
    R03_0[:, :, 3] = np.column_stack((x03_0[:, 1], y03_0[:, 3], z03_0[:, 1]))
    vecSW = vecSW / dSW
    mSWx = get_object_m(vecSW)
    R_psi = np.eye(3) + np.sin(psi) * mSWx + (1 - np.cos(psi)) * mSWx @ mSWx
    R03 = np.zeros((3, 3, 4))
    for i in range(4):
        R03[:, :, i] = R_psi @ R03_0[:, :, i]

    # 求解关节2
    theta2 = np.zeros(4)
    for i in range(4):
        theta2[i] = np.arcsin(-R03[1, 2, i])

    # 求解关节1
    theta1 = np.zeros(4)
    for i in range(4):
        if np.allclose(vecSW, np.array([0, 1, 0])):
            theta1[i] = 0
        else:
            theta1[i] = np.arctan2(R03[0, 2, i] , R03[2, 2, i] )

    # 求解关节3
    theta3 = np.zeros(4)
    for i in range(4):
        theta3[i] = np.arctan2(R03[1, 0, i] , R03[1, 1, i] )

    # 构造腕部姿态
    R34 = np.zeros((3, 3, 2))
    R34[:, :, 0] = rot_about_axis(theta4[0], 'x')
    R34[:, :, 1] = rot_about_axis(theta4[1], 'x')
    R43 = np.zeros((3, 3, 2))
    R43[:, :, 0] = R34[:, :, 0].T
    R43[:, :, 1] = R34[:, :, 1].T
    R30 = np.zeros((3, 3, 4))
    for i in range(4):
        R30[:, :, i] = R03[:, :, i].T
    R07 = mSW[:3, :3]
    R47 = np.zeros((3, 3, 4))
    R47[:, :, 0] = R43[:, :, 0] @ R30[:, :, 0] @ R07
    R47[:, :, 1] = R43[:, :, 1] @ R30[:, :, 1] @ R07
    R47[:, :, 2] = R43[:, :, 1] @ R30[:, :, 2] @ R07
    R47[:, :, 3] = R43[:, :, 0] @ R30[:, :, 3] @ R07

    # 求解关节6
    theta6 = np.zeros(4)
    for i in range(4):
        theta6[i] = np.arcsin(R47[2, 1, i])

    # 求解关节5
    theta5 = np.zeros(4)
    for i in range(4):
        theta5[i] = np.arctan2(-R47[0, 1, i], R47[1, 1, i])
    if abs(joint_limit[0, 4]) < joint_limit[1, 4]:
        theta5[theta5 < -2.61799388] += 2 * np.pi
    else:
        theta5[theta5 > 2.61799388] -= 2 * np.pi

    # 求解关节7
    theta7 = np.zeros(4)
    for i in range(4):
        theta7[i] = np.arctan2(-R47[2, 0, i], R47[2, 2, i])

    # 生成4组关节角
    joint_target_array = np.zeros((4, 7))
    joint_target_array[0, :] = [theta1[0], theta2[0], theta3[0], theta4[0], theta5[0], theta6[0], theta7[0]]
    joint_target_array[1, :] = [theta1[1], theta2[1], theta3[1], theta4[1], theta5[1], theta6[1], theta7[1]]
    joint_target_array[2, :] = [theta1[2], theta2[2], theta3[2], theta4[1], theta5[2], theta6[2], theta7[2]]
    joint_target_array[3, :] = [theta1[3], theta2[3], theta3[3], theta4[0], theta5[3], theta6[3], theta7[3]]


    # 限位判断
    theta_select = []
    for i in range(4):
        out_high_limit = np.any(joint_target_array[i, :] > joint_limit[1, :])
        out_low_limit = np.any(joint_target_array[i, :] < joint_limit[0, :])
        alpha = joint_target_array[i, 6]
        theta = joint_target_array[i, 5]
        d1,d2 = calculate_d1_d2(joint_limit, alpha, theta)
        if out_high_limit or out_low_limit:
            continue
        elif 155 <= d1 <= 195 and 155 <= d2 <= 195:
            theta_select.append(joint_target_array[i, :])

    if not theta_select:
        theta_best = theta_current
        error_info = 'all theta out of limit'
    else:
        theta_select = np.array(theta_select)
        dis_select = np.sum(np.abs(theta_select - theta_current), axis=1)
        num_select = np.argmin(dis_select)
        theta_best = theta_select[num_select, :]

    return theta_best, error_info

# 计算T28推杆行程
def calculate_d1_d2(joint_limit, alpha, theta):
    """
    参数:
    joint_limit: 机械臂限位
    alpha: 关节7关节角
    theta: 关节6关节角
    返回:
    d1,d2: 推杆行程
    """
    if joint_limit[1, 3] < abs(joint_limit[0, 3]):
        d1 = np.sqrt(
            (2250 * np.sin(-alpha) + 3900 * np.cos(alpha) * np.sin(-theta) - 17303 * np.cos(alpha) * np.cos(theta))**2 +
            (2250 * np.cos(alpha) + 17303 * np.sin(-alpha) * np.cos(theta) - 3900 * np.sin(-alpha) * np.sin(-theta) - 2250)**2 +
            (3900 * np.cos(theta) + 17303 * np.sin(-theta) - 2350)**2
        ) / 100
        d2 = np.sqrt(
            (2250 * np.sin(-alpha) - 3900 * np.cos(alpha) * np.sin(-theta) + 17303 * np.cos(alpha) * np.cos(theta))**2 +
            (2250 * np.cos(alpha) - 17303 * np.sin(-alpha) * np.cos(theta) + 3900 * np.sin(-alpha) * np.sin(-theta) - 2250)**2 +
            (3900 * np.cos(theta) + 17303 * np.sin(-theta) - 2350)**2
        ) / 100
    else:
        d1 = np.sqrt(
            (2250 * np.sin(alpha) + 3900 * np.cos(alpha) * np.sin(theta) - 17303 * np.cos(alpha) * np.cos(theta))**2 +
            (2250 * np.cos(alpha) + 17303 * np.sin(alpha) * np.cos(theta) - 3900 * np.sin(alpha) * np.sin(theta) - 2250)**2 +
            (3900 * np.cos(theta) + 17303 * np.sin(theta) - 2350)**2
        ) / 100
        d2 = np.sqrt(
            (2250 * np.sin(alpha) - 3900 * np.cos(alpha) * np.sin(theta) + 17303 * np.cos(alpha) * np.cos(theta))**2 +
            (2250 * np.cos(alpha) - 17303 * np.sin(alpha) * np.cos(theta) + 3900 * np.sin(alpha) * np.sin(theta) - 2250)**2 +
            (3900 * np.cos(theta) + 17303 * np.sin(theta) - 2350)**2
        ) / 100
    return d1, d2

# 计算机械臂的速度雅可比矩阵
def get_jacobian_velocity(para, angle, pos_tcp):
    """
    参数:
    para (numpy.ndarray): 运动学参数
    angle (numpy.ndarray): 关节角
    返回:
    jacob_velocity (numpy.ndarray): 速度雅可比矩阵
    """
    # 调用正运动学函数获取每个关节的位置和姿态
    t_jointi, rp = arm_forward_kinematics(angle, para, pos_tcp)
    # 初始化雅可比矩阵
    jacob_velocity = []
    # 关节轴方向
    axes = [1, 0, 2, 0, 2, 0, 1]

    for j in range(7):
        # 获取第 j 个关节的齐次变换矩阵
        rp_j = t_jointi[:, :, j]
        # 获取关节位置 P 和旋转矩阵 R
        pos = rp_j[:3, 3]
        rot = rp_j[:3, axes[j]]
        # 计算线速度部分 JvP 和角速度部分 JvR
        jacob_velocity_pos = rot
        jacob_velocity_rot = np.cross(rot, (rp[:3, 3] - pos))
        # 将线速度和角速度部分组合成一个 6x1 向量
        jacob_velocity_j = np.hstack((jacob_velocity_rot, jacob_velocity_pos))
        # 将当前关节的雅可比向量添加到总雅可比矩阵中
        jacob_velocity.append(jacob_velocity_j)

    # 将雅可比矩阵从列表转换为 NumPy 数组
    jacob_velocity = np.array(jacob_velocity).T
    return jacob_velocity

# 正运动学求解
def arm_forward_kinematics(angle, para, pos_tcp):
    """
    参数:
    Angle: 关节角
    para: 运动学参数
    返回:
    t_jointi: 各关节位姿矩阵
    m_tcp: 末端位姿矩阵
    """
    t = np.eye(4)
    t_jointi = np.zeros((4, 4, 7))
    axes = ['y', 'x', 'z', 'x', 'z', 'x', 'y']
    for i in range(7):
        idx = i * 6
        pos_i = para[idx:idx + 3]
        rpy_i = para[idx + 3:idx + 6]
        t_i = eul2tform(rpy_i, seq='XYZ')
        t_i[:3, 3] = pos_i

        r_rot = rot_about_axis(angle[i], axes[i])
        t_rot = np.eye(4)
        t_rot[:3, :3] = r_rot
        t_jointi[:, :, i] = t @ t_i
        t = t @ t_i @ t_rot

    t_tool = np.eye(4)
    t_tool[:3, 3] = pos_tcp
    m_tcp = t @ t_tool
    return t_jointi, m_tcp

# 求解绕指定轴旋转的旋转矩阵
def rot_about_axis(theta, axis):
    """
    参数:
    theta: 绕轴旋转的角度
    axis: 指定的轴方向
    返回:
    旋转矩阵
    """
    c = np.cos(theta)
    s = np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid axis: must be 'x', 'y', or 'z'")

# 将欧拉角转换为齐次变换矩阵
def eul2tform(euler_angles, seq='XYZ'):
    """
    参数:
    euler_angles: 欧拉角
    返回:
    tform: 齐次变换矩阵
    """
    rotation = R.from_euler(seq, euler_angles, degrees=False)
    tform = np.eye(4)
    tform[:3, :3] = rotation.as_matrix()
    return tform

# 位置+欧拉角转换为齐次变换矩阵
def rpy_pos2h(pos_rpy):
    """
    参数:
    pos_rpy: 位置+欧拉角
    返回:
    h: 齐次变换矩阵
    """
    pos = pos_rpy[:3]
    rpy = pos_rpy[3:]
    h = eul2tform(rpy, seq='XYZ')
    h[:3, 3] = pos
    return h

# 齐次变换矩阵转换为位置+欧拉角
def h2rpy_pos(h):
    """
    参数:
    h: 齐次变换矩阵
    返回:
    位置+欧拉角
    """
    rpy = R.from_matrix(h[:3, :3]).as_euler('XYZ')
    pos = h[:3, 3]
    return np.hstack((pos, rpy))

# 运动学参数转换为齐次变换矩阵
def para_cali2h(para_cali):
    """
    参数:
    para_cali: 运动学参数
    返回:
    h: 齐次变换矩阵
    """
    h = eul2tform(para_cali[3:], seq='XYZ')
    h[:3, 3] = para_cali[:3]
    return h

# 将旋转矩阵转换为轴角表示
def rotation_matrix_to_axis_angle(R):
    """
    参数:
    R (numpy.ndarray): 3x3 旋转矩阵
    返回:
    w (numpy.ndarray): 旋转轴（单位向量）乘以旋转角度
    """
    # 计算旋转角度
    theta = optimized_acos((np.trace(R) - 1) / 2)
    # 如果旋转角度为 0，旋转轴可以是任意单位向量
    if np.isclose(theta, 0):
        axis = np.array([1, 0, 0])  # 默认选择 x 轴
    else:
        # 计算旋转轴
        axis = (1 / (2 * np.sin(theta))) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        # 单位化旋转轴
        axis = axis / np.linalg.norm(axis)
    # 返回旋转轴和角度
    w = axis * theta
    return w

# 计算 arccos，确保输入值在 [-1, 1] 范围内，避免结果为复数
def optimized_acos(cos_theta):
    """
    参数:
    cos_theta (float or numpy.ndarray): 输入的余弦值
    返回:
    res (float or numpy.ndarray): 计算后的角度（弧度制）
    """
    # 将输入值限制在 [-1, 1] 范围内
    clipped_cos_theta = np.clip(cos_theta, -1, 1)
    # 计算 arccos
    res = np.arccos(clipped_cos_theta)
    return res

# 将向量 w 转换为反对称矩阵 m_w
def get_object_m(w):
    """
    参数:
    w (numpy.ndarray): 输入向量，形状为 (3,) 或 (3, 1)
    返回:
    m_w (numpy.ndarray): 反对称矩阵，形状为 (3, 3)
    """
    # 确保输入向量是 (3,) 或 (3, 1) 的形状
    w = np.atleast_1d(w).flatten()
    if len(w) != 3:
        raise ValueError("输入向量必须是长度为3的向量")
    # 创建反对称矩阵
    m_w = np.zeros((3, 3))
    m_w[0, 1] = -w[2]
    m_w[0, 2] = w[1]
    m_w[1, 0] = w[2]
    m_w[1, 2] = -w[0]
    m_w[2, 0] = -w[1]
    m_w[2, 1] = w[0]
    return m_w

# 计算两个区间集合的交集
def com_multi_interval(A, B):
    """
    参数:
    A (numpy.ndarray): 第一个区间集合，形状为 (2, n)，每一列表示一个区间
    B (numpy.ndarray): 第二个区间集合，形状为 (2, m)，每一列表示一个区间
    返回:
    res (numpy.ndarray): 交集后的区间集合，形状为 (2, k)，k 是交集后的区间数量
    """
    if A.size == 0 or B.size == 0:
        return np.array([])

    # 按照下限从小到大排序，若下限相同，则按上限从小到大排序
    A = A[:, np.argsort(A[0])]
    B = B[:, np.argsort(B[0])]

    res = []

    # 求取交集
    for m in range(A.shape[1]):
        for n in range(B.shape[1]):
            com_set = com_two_interval(A[:, m], B[:, n])
            if com_set.size > 0:
                res.append(com_set)

    if len(res) == 0:
        return np.array([])

    res = np.array(res).T
    return res

# 计算两个区间的交集
def com_two_interval(A, B):
    """
    参数:
    A (numpy.ndarray): 第一个区间，形状为 (2,)，表示 [下限, 上限]
    B (numpy.ndarray): 第二个区间，形状为 (2,)，表示 [下限, 上限]
    返回:
    res (numpy.ndarray): 交集区间，形状为 (2,)，如果无交集则返回空数组
    """
    if A.size == 0 or B.size == 0:
        return np.array([])

    if A[1] < B[0]:
        return np.array([])
    elif B[1] < A[0]:
        return np.array([])
    elif B[0] >= A[0] and B[0] <= A[1] and B[1] >= A[1]:
        return np.array([B[0], A[1]])
    elif A[0] >= B[0] and A[0] <= B[1] and A[1] >= B[1]:
        return np.array([A[0], B[1]])
    elif A[0] >= B[0] and A[0] <= B[1] and A[1] >= B[0] and A[1] <= B[1]:
        return A
    else:
        return B

# 计算多个区间的并集
def merge_intervals(intervals):
    """
    参数:
    intervals (numpy.ndarray): 区间数组，形状为 (2, n)，其中每一列表示一个区间 [下限, 上限]
    返回:
    res (numpy.ndarray): 并集后的区间数组，形状为 (2, m)，其中 m 是并集后的区间数量
    """
    if intervals.size == 0:
        return np.array([])

    # 按照下限从小到大排序，若下限相同，则按上限从小到大排序
    intervals = intervals.T
    intervals = intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]

    res = []
    current_interval = intervals[0]

    for i in range(1, intervals.shape[0]):
        if intervals[i, 0] <= current_interval[1]:
            current_interval[1] = max(current_interval[1], intervals[i, 1])
        else:
            res.append(current_interval)
            current_interval = intervals[i]

    res.append(current_interval)
    res = np.array(res).T

    return res