
from launch import LaunchDescription
from launch_ros.actions import Node

"""
    How to use Node in launch
"""

def generate_launch_description():
    turtle1 = Node(
        package="turtlesim",
        executable="turtlesim_node"
    )
    return LaunchDescription([turtle1])