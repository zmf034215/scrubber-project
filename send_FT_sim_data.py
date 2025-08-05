import lcm
from lcm_data_structure.ecat_debug_ft_data_lcmt import ecat_debug_ft_data_lcmt
import time
def send_ft_data():
    # 初始化LCM
    lc = lcm.LCM()
    
    # 创建力传感器数据对象
    ft_data = ecat_debug_ft_data_lcmt()
    
    # 设置第三个数据（original_Fz）为10.0
    # 注意：由于数据是列表类型，需要通过索引修改
    ft_data.original_Fz[0] = 10.0
    
    # 其他数据保持默认初始值0.0，也可根据需要设置
    # ft_data.original_Fx[0] = 0.0  # 默认值
    # ft_data.original_Fy[0] = 0.0  # 默认值
    # ft_data.original_Mx[0] = 0.0  # 默认值
    # ft_data.original_My[0] = 0.0  # 默认值
    # ft_data.original_Mz[0] = 0.0  # 默认值
    
    # 发送LCM消息，主题名称可根据实际需求修改

    while True:
            # 记录发送开始时间
        start_time = time.time()
        
        # 发送LCM消息
        lc.publish("ecat_debug_FT_dataARM_FT_L", ft_data.encode())
        
        # 计算需要等待的时间（4ms = 0.004秒）
        elapsed = time.time() - start_time
        sleep_time = 0.004 - elapsed
        
        # 确保等待时间非负
        if sleep_time > 0:
            time.sleep(sleep_time)
            


if __name__ == "__main__":
    send_ft_data()