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
from std_msgs.msg import Float64

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

        # argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", default=42, type=int, help='Sets Gym seed')

        parser.add_argument("--save_root_path", default='/src/mixed_reality_py/mixed_reality_py/results/sac/train/', type=str, help='Sets Gym seed')
        parser.add_argument("--obs_dim", default=5, type=int, help='Manually set')
        parser.add_argument("--action_dim", default=2, type=int, help='Manually set')
        parser.add_argument(
            "--max_episode_steps",
            default=4e3,
            type=int,
            help='Max number of steps to run environment')
        parser.add_argument(
            "--alpha",
            default=0.2,
            type=float,
            help=
            'Determines the relative importance of entropy term against the reward'
        )
        args = parser.parse_args()

        # real car <---> virtual car2
        # pose initialisation 
        self.pose_x_virtual_car2 = 0.0
        self.pose_y_virtual_car2 = 0.0
        self.pose_theta_virtual_car2 = 0.0

        """
            Publishers for virtual car1
        """
        self.pub_scan = self.create_publisher(LaserScan, "/virtual_car1/scan", 10)
        self.pub_odom = self.create_publisher(Odometry, "/virtual_car1/odom", 10)
        self.pub_goal = self.create_publisher(PointStamped, "/virtual_car1/goal", 10)
        # transform broadcaster
        self.tf_br = TransformBroadcaster(self) 

        """
            Subscribers from real car then cloned to virtual car2
        """
        # self.pose_sub = self.create_subscription(PoseStamped, "/pf/viz/inferred_pose", self.pose_callback, 10)
        odom_topic = 'odom'
        lidarscan_topic = 'scan'
        pf_pose_topic = '/pf/viz/inferred_pose'
        steer_topic = '/real_car/steering_angle'
        sub_odom = message_filters.Subscriber(self, Odometry, odom_topic)
        sub_scan = message_filters.Subscriber(self, LaserScan, lidarscan_topic)
        sub_pose = message_filters.Subscriber(self, PoseStamped, pf_pose_topic)
        sub_steer = message_filters.Subscriber(self, Float64, steer_topic)

        # laser scan        
        scan_fov = 4.7
        scan_beams = 1080
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        # distance laser to base_link is (1/2*car_height + 1/2*lidar_height)=0.05+0.025=0.075
        self.dist_laser2baselink = 0.075

        # map initialisation
        with open(path + '/src/mixed_reality_py/mixed_reality_py/new_map/DEU116_map_closing.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        # create env
        self.env = F110Env(map=path + conf.map_path, map_ext= conf.map_ext, num_agents=2, seed=args.seed)

        # Initialize model, algorithm, agent, replay_memory
        model = ACModel(FRAMES, args.obs_dim, args.action_dim)
        algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=args.alpha, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

        # 2 cars - 2 agents
        self.agent_1 = SACAgent(algorithm)
        self.agent_2 = SACAgent(algorithm)

        # load best model
        best_model_path = path + args.save_root_path + str(args.seed) + '/model/sac_last.ckpt'
        # 2 separated controllers to control 2 agents
        self.agent_1.restore(best_model_path)
        self.agent_2.restore(best_model_path)

        # lidar scan for agents
        self.scans_stack_1 = deque(maxlen=FRAMES)
        self.scans_stack_2 = deque(maxlen=FRAMES)
        
        # reset env
        initial_list = initialize_episode2()
        obs, step_rewards, done, self.info, position_goals = self.env.reset(initial_list)

        # pose and lidar scan for car1 - index is 0
        self.pose_x = obs['poses_x'][0]
        self.pose_y = obs['poses_y'][0]
        self.pose_theta = obs['poses_theta'][0]
        self.linear_vel_x = obs['linear_vels_x'][0]
        self.lidar_scan_range = list(np.array(obs['scans'][0], dtype=float))

        # targets to reach for 2 agents
        target_1_x = position_goals[0][0]
        target_1_y = position_goals[0][1]
        target_2_x = position_goals[1][0]
        target_2_y = position_goals[1][1]

        # goal position for car1
        self.goal_x_car1 = target_1_x
        self.goal_y_car1 = target_1_y
        self.goal_x_car2 = target_2_x
        self.goal_y_car2 = target_2_y

        # 3 frames of scans from observations appended into stack for 2 agents
        self.scans_stack_1.append(obs['scans'][0])
        self.scans_stack_1.append(obs['scans'][0])
        self.scans_stack_1.append(obs['scans'][0])
        self.scans_stack_2.append(obs['scans'][1])
        self.scans_stack_2.append(obs['scans'][1])
        self.scans_stack_2.append(obs['scans'][1])

        # states for car 1
        self.scans_1 = copy.deepcopy(self.scans_stack_1)
        self.state_1 = np.array([
                            target_1_x - obs['poses_x'][0], 
                            target_1_y - obs['poses_y'][0],
                            obs['poses_theta'][0],
                            obs['linear_vels_x'][0],
                            obs['ang_vels_z'][0]
                            ])
        # states for car 2
        self.scans_2 = copy.deepcopy(self.scans_stack_2)
        self.state_2 = np.array([
                            target_2_x - obs['poses_x'][1], 
                            target_2_y - obs['poses_y'][1],
                            obs['poses_theta'][1],
                            obs['linear_vels_x'][1],
                            obs['ang_vels_z'][1]
                            ])
        
        # initialisation of episode reward for 2 agents
        self.episode_reward_1 = 0
        self.episode_reward_2 = 0

        # initialisation of step
        self.step = 0

        # time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer([sub_odom, sub_pose], 1000, 0.1)
        ts.registerCallback(self.realCar_callback)


    def realCar_callback(self, odom, pose):
        print('--------------------------------------------')
        self.step = self.step + 1
        self.pose_x_virtual_car2 = pose.pose.position.x
        self.pose_y_virtual_car2 = pose.pose.position.y
        # self.pose_theta_virtual_car2 = tf_transformations.euler_from_quaternion([pose_msg.pose.orientation.x,
        #                                                                          pose_msg.pose.orientation.y,
        #                                                                          pose_msg.pose.orientation.z,
        #                                                                          pose_msg.pose.orientation.w])[2]
        self.pose_theta_virtual_car2 = math.radians(
                                            R.from_quat([pose.pose.orientation.x, 
                                                         pose.pose.orientation.y,
                                                         pose.pose.orientation.z,
                                                         pose.pose.orientation.w]).as_euler('zyx', degrees=True)[0])
        print("pose_x_virtual_car2 is:", self.pose_x_virtual_car2)
        print("pose_y_virtual_car2 is:", self.pose_y_virtual_car2)
        print("pose:", self.pose_theta_virtual_car2)


        # action for agent1
        action_1 = self.agent_1.predict(self.scans_1, FRAMES, self.state_1)
        action_1 = maneuver(action_1, self.info['reach_cond_1'], self.info['collision_1'])

        # action for agent2
        self.state_2 = np.array([
                            self.pose_x_virtual_car2, 
                            self.pose_y_virtual_car2,
                            self.pose_theta_virtual_car2,
                            odom.twist.twist.linear.x,
                            odom.twist.twist.angular.z
                            ])
        # action_2 = self.agent_2.predict(self.scans_2, FRAMES, self.state_2)
        # action_2 = list(maneuver(action_2, self.info['reach_cond_2'], self.info['collision_2']))
        action_2 = [0.0, 0.0]

        # s' for 2 agents    
        obs_prime, step_rewards, done, self.info = \
            self.env.step(np.array([action_1, action_2]), [self.state_1, self.state_2], reset_flag=False)

        """
            Publish scan and pose of virtual car1
        """
        timestamp = self.get_clock().now().to_msg()
        # self.pub_pose_virtualCar1(timestamp)
        self.pub_scan_virtualCar1(timestamp)
        self.pub_odom_virtualCar1(timestamp)
        self.pub_goal_virtualCar1(timestamp)

        # pub tf of car1
        self.pub_tf_baselink2odom(timestamp)
        self.pub_tf_laser2baselink(timestamp)

        # car1 observation s'
        self.pose_x = obs_prime['poses_x'][0]
        self.pose_y = obs_prime['poses_y'][0]
        self.pose_theta = obs_prime['poses_theta'][0]
        self.linear_vel_x = obs_prime['linear_vels_x'][0]
        self.lidar_scan_range = list(np.array(obs_prime['scans'][0], dtype=float))

        # s' for agent1
        self.scans_stack_1.append(obs_prime['scans'][0])
        state_prime_1 = np.array([self.goal_x_car1 - obs_prime['poses_x'][0],  self.goal_y_car1 - obs_prime['poses_y'][0],
                                  obs_prime['poses_theta'][0],
                                  obs_prime['linear_vels_x'][0],
                                  obs_prime['ang_vels_z'][0]])
        # s' for agent2
        self.scans_stack_2.append(obs_prime['scans'][1])
        state_prime_2 = np.array([self.goal_x_car2 - self.pose_x_virtual_car2,  self.goal_y_car2 - self.pose_y_virtual_car2,
                                  self.pose_theta_virtual_car2,
                                  odom.twist.twist.linear.x,
                                  odom.twist.twist.angular.z])

        # s = s' for 2 agents
        self.scans_1 = copy.deepcopy(self.scans_stack_1)
        self.state_1 = state_prime_1
        self.scans_2 = copy.deepcopy(self.scans_stack_2)
        self.state_2 = state_prime_2
            
        # environment rendering 
        self.env.render(mode='human')

        # print('7. done info:', info)
        if not done:
            # episode rewards for 2 agents
            self.episode_reward_1 += step_rewards[0][0]
            self.episode_reward_2 += step_rewards[1][0]
        print("hello")


    def pub_pose_virtualCar1(self, timestamp):
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = "virtual_car1_base_link"
        pose.pose.position.x = float(self.pose_x)
        pose.pose.position.y = float(self.pose_y)
        quaternion = quaternion_from_euler(0., 0., self.pose_theta)
        pose.pose.orientation.w = quaternion[0]
        pose.pose.orientation.x = quaternion[1]
        pose.pose.orientation.y = quaternion[2]
        pose.pose.orientation.z = quaternion[3]
        self.pub_pose.publish(pose)

    def pub_scan_virtualCar1(self, timestamp):
        scan = LaserScan()
        scan.header.stamp = timestamp
        scan.header.frame_id = "virtual_car1_laser"
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.lidar_scan_range
        self.pub_scan.publish(scan)

    def pub_odom_virtualCar1(self, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "virtual_car1_base_link"
        odom.pose.pose.position.x = float(self.pose_x)
        odom.pose.pose.position.y = float(self.pose_y)
        quat = quaternion_from_euler(0., 0., self.pose_theta)
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]
        odom.pose.pose.orientation.w = quat[0]
        odom.twist.twist.linear.x = float(self.linear_vel_x)
        odom.twist.twist.linear.y = 0.
        odom.twist.twist.angular.z = 0.
        self.pub_odom.publish(odom)

    def pub_tf_baselink2odom(self, timestamp):
        tf = Transform()
        tf.translation.x = float(self.pose_x)
        tf.translation.y = float(self.pose_y)
        tf.translation.z = 0.0
        quat = quaternion_from_euler(0., 0., self.pose_theta)
        tf.rotation.x = quat[1]
        tf.rotation.y = quat[2]
        tf.rotation.z = quat[3]
        tf.rotation.w = quat[0]

        tf_stamped = TransformStamped()
        tf_stamped.transform = tf
        tf_stamped.header.stamp = timestamp
        tf_stamped.header.frame_id = "map"
        tf_stamped.child_frame_id = "virtual_car1_base_link"
        self.tf_br.sendTransform(tf_stamped)

    def pub_tf_laser2baselink(self, timestamp):
        tf = TransformStamped()
        tf.transform.translation.x = self.dist_laser2baselink
        tf.transform.rotation.w = 1.
        tf.header.stamp = timestamp
        tf.header.frame_id = "virtual_car1_base_link"
        tf.child_frame_id = "virtual_car1_laser"
        self.tf_br.sendTransform(tf)

    def pub_goal_virtualCar1(self, timestamp):
        goal = PointStamped()
        goal.header.stamp = timestamp
        goal.header.frame_id = "map"
        goal.point.x = float(self.goal_x_car1)
        goal.point.y = float(self.goal_y_car1)
        goal.point.z = 0.
        self.pub_goal.publish(goal)



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