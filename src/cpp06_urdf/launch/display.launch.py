from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    cpp06_urdf_dir = get_package_share_directory("cpp06_urdf")
    default_model_path = os.path.join(cpp06_urdf_dir, "urdf/urdf", "demo01_helloworld.urdf")
    default_rviz_path = os.path.join(cpp06_urdf_dir, "rviz", "f110_localisation.rviz")
    model = DeclareLaunchArgument(name="model", default_value=default_model_path)

    # 1. Start robot_state_publisher node that loads urdf file as parameter
    p_value = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]))
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description":p_value}]      
    )

    # 2. Start joint_state_publisher - publish states of non-fixed joints
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher"
    )
    
    # 3. Start rviz2
    rviz2 = Node(
        package="rviz2", 
        executable="rviz2",
        arguments=["-d", default_rviz_path]
        )

    # return LaunchDescription([model, robot_state_pub, rviz2])
    return LaunchDescription([model, robot_state_pub, joint_state_publisher, rviz2])