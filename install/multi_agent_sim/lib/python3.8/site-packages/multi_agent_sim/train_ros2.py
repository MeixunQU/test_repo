"""
    This script is used to start RL training process.
    To be integrated into ROS2 system, the original codes should be adpated to the ROS2 templates.
    
    The parameters(server/client) function is used to load hyper-parameters from server. 
"""

# 1. import packages
# 1.1 ROS2 related packages
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import ListParameters
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterValue
from ros2param.api import get_parameter_value

# 1.2 RL training - evn, agent model, algorithm
from multi_agent_sim.env_sac_f110 import F110Env
from multi_agent_sim.sac_naked import *
from multi_agent_sim.initialize import initialize_episode

import numpy as np
import torch
from parl.utils import logger, summary

# 1.3 path
import os
import sys
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/multi_agent_sim/multi_agent_sim")

# 1.4 others
import yaml
import argparse
from argparse import Namespace
import random
from collections import deque
import math


# 3. define Node class
class RLtraining(Node):
    def __init__(self):
        super().__init__('RL_training_process')

    def list_params(self):
        # 3-1.create cient
        cli_list = self.create_client(ListParameters, '/RL_training_params_server/list_parameters')
        # 3-2.wait for the service connection
        while not cli_list.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Listing the parameters, please wait...')
        req = ListParameters.Request()
        future = cli_list.call_async(req)
        rclpy.spin_until_future_complete(self,future)
        return future.result()

    def get_params(self,names):
        # 3-1.create client
        cli_get = self.create_client(GetParameters, '/param_server_node_py/get_parameters')
        # 3-2.wait for the service connection
        while not cli_get.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Listing the parameters, please wait...')
        req = GetParameters.Request()
        req.names = names
        future = cli_get.call_async(req)
        rclpy.spin_until_future_complete(self,future)
        return future.result()


def main():
    # 2. ROS2 Node initialisation
    rclpy.init()

    # 4. create object
    RL_training_client = RLtraining()

    # get params list
    RL_training_client.get_logger().info("---------get params list---------")

    response = RL_training_client.list_params()
    for name in response.result.names:
        RL_training_client.get_logger().info(name)

    RL_training_client.get_logger().info("---------get params---------")
    names = ["height","car_type", "width"]
    response = RL_training_client.get_params(names)
    # print(response.values)
    for v in response.values:
        if v.type == ParameterType.PARAMETER_STRING:
            RL_training_client.get_logger().info("string:%s" % v.string_value)
        elif v.type == ParameterType.PARAMETER_DOUBLE:
            RL_training_client.get_logger().info("float:%.2f" % v.double_value)

    rclpy.shutdown()



if __name__ == '__main__':
    main()
