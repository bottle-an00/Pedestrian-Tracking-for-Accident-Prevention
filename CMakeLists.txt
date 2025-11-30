cmake_minimum_required(VERSION 3.0.2)
project(pedestrian_tracking)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

# 파이썬 설정을 먼저 호출합니다.
catkin_python_setup()

## Declare ROS messages
add_message_files(
  FILES
  GpsInfo.msg
  DetectionInfo.msg
  PipelinePacket.msg
)

## Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime
)
