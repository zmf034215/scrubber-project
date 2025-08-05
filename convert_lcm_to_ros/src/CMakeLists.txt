cmake_minimum_required(VERSION 3.5)
project(lcm2ros)  # 需与package.xml中的名称一致

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 寻找依赖
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# 安装Python模块
ament_python_install_package(${PROJECT_NAME})

# 安装可执行脚本（指定你的Python脚本路径）
install(PROGRAMS
  scripts/lcm2ros.py  
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()