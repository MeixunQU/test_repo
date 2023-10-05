
from pkg_resources import declare_namespace
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument

"""
    Start train node
"""

def generate_launch_description():
    # multi_agent_sim = get_package_share_directory("multi_agent_sim")
    # default_train_path = os.path.join(multi_agent_sim, "multi_agent_sim", "train.py")
    # start_train = DeclareLaunchArgument(name="train", default_value=default_train_path)
    
    # start_train = Node(
    #     package="multi_agent_sim",
    #     executable="train",
    #     exec_name="RL_training_start",
    #     parameters=[{"seed":1234, 
    #                  "save_root_path":'./src/multi_agent_sim/multi_agent_sim/results/sac/train/',
    #                  "obs_dim":5,
    #                  "action_dim":2,
    #                  "train_total_episodes":2500,
    #                  "max_episode_step":12e2,
    #                  "test_episode_interval":10,
    #                  "test_episode_number":5,
    #                  "alpha":0.2
    #                  }],
    #     respawn=False
    # )

    decl_seed = DeclareLaunchArgument(name="seed", default_value='12345')
    decl_save_root_path = DeclareLaunchArgument(name="save_root_path", default_value='./src/multi_agent_sim/multi_agent_sim/results/sac/train/')
    decl_obs_dim = DeclareLaunchArgument(name="obs_dim", default_value='5')
    decl_action_dim = DeclareLaunchArgument(name="action_dim", default_value='2')
    decl_train_episodes = DeclareLaunchArgument(name="train_total_episodes", default_value='2500')
    decl_max_step = DeclareLaunchArgument(name="max_episode_steps", default_value='12e2')
    decl_test_interval = DeclareLaunchArgument(name="test_episode_interval", default_value='10')
    decl_test_number = DeclareLaunchArgument(name="test_episode_number", default_value='5')
    decl_alpha = DeclareLaunchArgument(name="alpha", default_value='0.2')

    start_train = Node(
        package="multi_agent_sim",
        executable="train",
        parameters=[{"seed": LaunchConfiguration("seed"),
                     "save_root_path": LaunchConfiguration("save_root_path"),
                     "obs_dim": LaunchConfiguration("obs_dim"),
                     "action_dim": LaunchConfiguration("action_dim"),
                     "train_total_episodes": LaunchConfiguration("train_total_episodes"),
                     "max_episode_step": LaunchConfiguration("max_episode_steps"),
                     "test_episode_interval": LaunchConfiguration("test_episode_interval"),
                     "test_episode_number": LaunchConfiguration("test_episode_number"),
                     "alpha": LaunchConfiguration("alpha")
                     }]
    )

    return LaunchDescription([decl_seed, decl_save_root_path, decl_obs_dim, decl_action_dim, decl_train_episodes, decl_max_step, decl_test_interval,
                              decl_test_number, decl_alpha, start_train])