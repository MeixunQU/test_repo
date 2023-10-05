"""
   Subscribe: position(x,y), pose_theta(yaw_angle), velocity(linear_vels_x), angular_velocity_z,
              lidar_scan
   
   Publish: steering_angle and speed  
"""
# 1. import packages
# 1.1 ros packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import message_filters
import tf_transformations

# 1.2 controller - paddle inference model
import argparse
import numpy as np
import paddle.inference as paddle_infer
import pickle

# 1.3 others
import math
import os


# read model file and params file (static)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default='./src/f110_topic_py/f110_topic_py/inference_model/actor.pdmodel', type=str, help="model filename")
    parser.add_argument("--params_file", default='./src/f110_topic_py/f110_topic_py/inference_model/actor.pdiparams', type=str, help="parameter filename")

    return parser.parse_args()


# clip steering angle and velocity
def maneuver(action):
    steer_limit = 8
    speed_upper_limit = 1.3
    speed_lower_limit = 0.3
    
    action[0] = action[0] * np.pi / steer_limit  # steering angle
    action[1] = abs(action[1] * speed_upper_limit)  # velocity
    if action[0] < -np.pi / steer_limit:
        action[0] = -np.pi / steer_limit
    if action[0] > np.pi / steer_limit:
        action[0] = np.pi / steer_limit
    if action[1] > speed_upper_limit:
        action[1] = speed_upper_limit
    if action[1] < speed_lower_limit:
        action[1] = speed_lower_limit

    return action


class SACcontroller(Node):
    def __init__(self):
        super().__init__('sac_controller_node')

        # fixed target positions
        self.target_x = -0.24963176250457764
        self.target_y = 6.710972785949707

        """
            3. instantiate paddle inference model
        """
        args = parse_args()
        # 3.1 create config
        config = paddle_infer.Config(args.model_file, args.params_file)
        # 3.2 create predictor with config
        self.predictor = paddle_infer.create_predictor(config)
        # 3.3 get input names and handles of scans and obs
        input_names = self.predictor.get_input_names()
    
        self.scans_input_handle = self.predictor.get_input_handle(input_names[0])
        self.scans_input_handle.reshape([1, 3, 1080])  # last 3 frames of scan data

        self.obs_input_handle = self.predictor.get_input_handle(input_names[1])
        self.obs_input_handle.reshape([1, 5])

        self.obs_input = None

        # smooth steering angle 
        self.prev_steering = 0.0

        # three frames of scan data
        self.count = 0 # laser scan counter
        self.scan_frame3 = None
        self.scan_frame2 = None
        self.scan_frame1 = None
        self.scan_frame_new = None
        self.laser_scan_input = None
        # position x, y
        self.position_x = 0.0
        self.position_y = 0.0
        # angular_velocity_z
        self.ang_vels_z = 0.0
        # linear velocity x
        self.linear_vel_x = 0.0

        # smooth steering angle
        # self.steering_angle_1 = 0.0
        # self.steering_angle_2 = 0.0
        # self.steering_angle_3 = 0.0
        # self.steering_angle_4 = 0.0
        # self.steering_angle_5 = 0.0
        
        """
            4. publisher
        """
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "drive", 10)


        """
            5. subscriber & synchronizer
        """
        odom_topic = 'odom'
        lidarscan_topic = 'scan'
        pf_pose_topic = '/pf/viz/inferred_pose'
        sub_odom = message_filters.Subscriber(self, Odometry, odom_topic)
        sub_scan = message_filters.Subscriber(self, LaserScan, lidarscan_topic)
        sub_pose = message_filters.Subscriber(self, PoseStamped, pf_pose_topic)

        # time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer([sub_odom, sub_scan, sub_pose], 1000, 0.1)
        ts.registerCallback(self.sac_callback)

    """
        6. callback func
    """
    def sac_callback(self, odom, scan, pose):
        """
            6.1 scan msg - 3 frames
        """
        # print(type(scan.ranges)) # array.array
        self.scan_frame_new = np.array(scan.ranges)   # 1080 beams scan ranges
        if self.count == 0:
            self.scan_frame3 = self.scan_frame_new
            self.scan_frame2 = self.scan_frame_new
            self.scan_frame1 = self.scan_frame_new
            # print(self.scan_frame3)
            # print(self.scan_frame2)
            self.laser_scan_input = np.array([[self.scan_frame3, self.scan_frame2, self.scan_frame1]])
            # print(self.laser_scan_input)
            self.count += 1
        else:
            self.scan_frame1 = self.scan_frame_new
            self.laser_scan_input = np.array([[self.scan_frame3, self.scan_frame2, self.scan_frame1]])
            # print(self.laser_scan_input)
            self.scan_frame3 = self.scan_frame2
            self.scan_frame2 = self.scan_frame1
            self.count += 1

        """
            6.2 position(x,y)
        """
        self.position_x = pose.pose.position.x
        self.position_y = pose.pose.position.y

        """
            6.3 linear velocity x
        """
        self.linear_vel_x = odom.twist.twist.linear.x

        """
            6.4 yaw angle (heading angle)
        """
        self.yaw_angle = tf_transformations.euler_from_quaternion([pose.pose.orientation.x,
                                                                   pose.pose.orientation.y,
                                                                   pose.pose.orientation.z,
                                                                   pose.pose.orientation.w])[2]
        """
            6.5 angular_velocity_z
        """
        self.ang_vels_z = odom.twist.twist.angular.z

        """
            7. organize obs_input
        """
        self.obs_input = np.array([[self.target_x-self.position_x, self.target_y-self.position_y, self.yaw_angle, self.linear_vel_x, self.ang_vels_z]])
        self.obs_input = self.obs_input.astype('float32')

        # scan & obs input handle
        self.laser_scan_input = self.laser_scan_input.astype('float32')
        self.scans_input_handle.copy_from_cpu(self.laser_scan_input)
        self.obs_input_handle.copy_from_cpu(self.obs_input)

        """
            8. run sac model and publish steering angle & speed
        """
        # run predictor
        self.predictor.run()

        # output
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        action = output_handle.copy_to_cpu()[0]
        action = np.tanh(action)
        action = maneuver(action) # clip

        # smooth steering angle
        # self.steering_angle_1 = action[0]
        # self.steering_angle_2 = self.steering_angle_1
        # self.steering_angle_3 = self.steering_angle_2
        # self.steering_angle_4 = self.steering_angle_3
        # self.steering_angle_5 = self.steering_angle_4
        # steering_angle_mean = np.mean(np.array([
        #                                 self.steering_angle_1,
        #                                 self.steering_angle_2,
        #                                 self.steering_angle_3,
        #                                 self.steering_angle_4,
        #                                 self.steering_angle_5]))

        # print info
        # self.get_logger().info(f'States: {self.position_x}, {self.position_y}, {self.yaw_angle}, {self.linear_vel_x}, {self.ang_vels_z}')
        self.get_logger().info(f'predict action (static): {action[0]}, {action[1]}')
        # self.get_logger().info(f'mean steering_angle (last 5 steps):{steering_angle_mean}')

        # publish steering angle and speed
        drive_msg = AckermannDriveStamped()
        now_time = self.get_clock().now()
        drive_msg.header.stamp = now_time.to_msg()
        drive_msg.header.frame_id = "base_link"

        # reach the goal position
        target_pos = [self.target_x, self.target_y]
        curr_pos = [self.position_x, self.position_y]
        dist = math.dist(curr_pos, target_pos)
        
        # conditions to stop
        # 1. distance between current position and target position < 0.35 meter
        if dist < 0.35: 
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.pub_drive.publish(drive_msg)
        else:
            drive_msg.drive.steering_angle = float(action[0])
            # drive_msg.drive.steering_angle = float(steering_angle_mean)  # mean steering angle for last 5 steps
            drive_msg.drive.speed = float(action[1])
            self.pub_drive.publish(drive_msg)
        # 2. distance to obstacles is too close
        # 3. time out

        # self.get_logger().info(f'Distance to goal: {dist}')
        # self.get_logger().info(f'steering_angle and speed: {drive_msg.drive.steering_angle}, {drive_msg.drive.speed}')
        print('----------------------------------------------------------------------------')

def main():

    # 2. ROS node initialisation
    rclpy.init()

    # 3. spin() with SACcontroller(node class) input
    controller = SACcontroller()
    rclpy.spin(controller)
    
    # 9. release resource
    rclpy.shutdown()

if __name__ == '__main__':
    main()