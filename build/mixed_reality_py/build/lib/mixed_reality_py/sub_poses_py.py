"""
    Virtual car 1: 
            Publish: pose(position x, position y, yaw angle), scan
            Aim: publish pose and scan to make RViz to visualise the virtual car's moving track
    
    Virtual car 2 (real car's twin):
            Subscribe(from real car): pose(particle filter), linear_vel_x (odometry), angular_vel_z (yaw rate)
            Publish(from virtual car): scan with regard to the pose, linear_vel_x, yaw rate from real car
"""

# 1. import packages
# 1.1 ros packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from sensor_msgs.msg import JointState
import message_filters
import tf_transformations
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PointStamped

# 1.2 path
import numpy as np
import math
import os
import sys
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/mixed_reality_py/mixed_reality_py")

# 1.3 f1tenth_gym simulator
from mixed_reality_py.env_sac_f110 import F110Env
import yaml
import argparse
from argparse import Namespace
import numpy as np
from collections import deque
from sac_naked import *
from mixed_reality_py.initialize import initialize_episode2
import random
from scipy.spatial.transform import Rotation as R


# clip steering angle and velocity for multiple cars
def maneuver(action, reach_flag, collision_flag):
    if reach_flag or collision_flag:
        action[0] = 0
        action[1] = 0
    else:
        action[0] = action[0] * np.pi / 8  # steering angle
        action[1] = abs(action[1] * 4)  # velocity
        if action[0] < -np.pi / 8:
            action[0] = -np.pi / 8
        if action[0] > np.pi / 8:
            action[0] = np.pi / 8
        if action[1] > 1.3:
            action[1] = 1.3

    return action

# function to convert the euler to quaternion
def quaternion_from_euler(roll, pitch, yaw):
    """
       Converts euler roll, pitch, yaw to quaternion (w in last place)
       quat = [w, x, y, z]
       Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q

# Global variables
WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
FRAMES = 3

class MultiAgentController(Node): 

    def __init__(self):
        super().__init__('multi_agent_sac_controller_node')
        """
            Subscribers from real car then cloned to virtual car2
        """
        self.pose_sub = self.create_subscription(PoseStamped, "/pf/viz/inferred_pose", self.pose_callback, 10)



    def pose_callback(self, msg):
        a = msg.pose.position.x 
        return a 



def main():

    # 2. ROS node initialisation
    rclpy.init()

    # 3. spin() with MultiAgentController(node class) input
    controller = MultiAgentController()
    rclpy.spin(controller)

    # 9. realease resource
    rclpy.shutdown()


if __name__ == '__main__':
    main()