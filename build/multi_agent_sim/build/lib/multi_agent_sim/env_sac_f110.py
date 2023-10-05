# gym imports
import gym
# from gym import error, spaces, utils
# from gym.utils import seeding

# base classes
from multi_agent_sim.base_classes import Simulator
from multi_agent_sim.lidar_process import lidar_analysis
# others
import numpy as np
import os
import time
# import random
import math
# gl
import pyglet

pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:

            # Hong rui's params
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                           'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
                           'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31,
                           'length': 0.58}

            # params from 5 traces per round for No.10 training result (15 rounds)
            # self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.16816832, 'lr': 0.15177377, 'h': 0.074,
            #                'm': 3.74, 'I': 0.04712, 's_min': -0.52231318, 's_max': 0.51628776, 'sv_min': -3.42537148, 'sv_max': 3.88872181,
            #                'v_switch': 8.53771819, 'a_max': 9.6069614, 'v_min': -4.94960142, 'v_max': 19.96697248, 'width': 0.31,
            #                'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

        self.target_position = []
        self.target_goal = []
        self.config = {"reward_weights": [1, 1, 0, 0, 0]}
        self.achieved_goal = []
        self.error_turn_num = 0
        # self.bubble_radius = 1.5
        self.bubble_radius = 0.6  # DEU116_map

        self.last_avg_distance = self.bubble_radius


    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout/episode is done

        Args:
            None

        Returns:
            done (bool): whether the rollout/episode is done
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        # left_t = 2
        # right_t = 2
        #
        # poses_x = np.array(self.poses_x) - self.start_xs
        # poses_y = np.array(self.poses_y) - self.start_ys
        # delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        # temp_y = delta_pt[1, :]
        # idx1 = temp_y > left_t
        # idx2 = temp_y < -right_t
        # temp_y[idx1] -= left_t
        # temp_y[idx2] = -right_t - temp_y[idx2]
        # temp_y[np.invert(np.logical_or(idx1, idx2))] = 0
        #
        # dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        # closes = dist2 <= 0.1
        # for i in range(self.num_agents):
        #     if closes[i] and not self.near_starts[i]:
        #         self.near_starts[i] = True
        #         self.toggle_list[i] += 1
        #     elif not closes[i] and self.near_starts[i]:
        #         self.near_starts[i] = False
        #         self.toggle_list[i] += 1
        #     self.lap_counts[i] = self.toggle_list[i] // 2
        #     if self.toggle_list[i] < 4:
        #         self.lap_times[i] = self.current_time

        # done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        # position goal
        reach_cond = False
        position_achieved = np.array([self.achieved_goal[0], self.achieved_goal[1]])
        position_goal = np.array(self.target_position[0])
        position_cond = np.abs(position_achieved - position_goal)
        # print('position reach condition:', position_cond)

        velocity_achieved = np.array([self.achieved_goal[3]])
        velocity_goal = np.array([self.target_goal[3]])
        velocity_cond = np.abs(velocity_achieved - velocity_goal)
        # print('velocity reach condition:', velocity_cond)

        if position_cond[0] < 0.4 and position_cond[1] < 0.4:
            # if velocity_cond.any() == 0:
            reach_cond = True
                # print('reach condition:', reach_cond)

        done = (self.collisions[self.ego_idx]) or reach_cond

        # return done, self.toggle_list >= 4
        return done

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action, last_state=[], reset_flag=True):
    # def step(self, episode_num, action, last_state=[], reset_flag=True):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # episode_num = episode_num
        # print("In step function, episode_num is:", episode_num)

        # call simulation step
        # print('1. action:', action)
        obs = self.sim.step(action)
        # print('3. obs:', obs)
        # print('----------------------')
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        F110Env.current_obs = obs

        # self.render_obs = {
        #     'ego_idx': obs['ego_idx'],
        #     'poses_x': obs['poses_x'],
        #     'poses_y': obs['poses_y'],
        #     'poses_theta': obs['poses_theta'],
        #     'lap_times': obs['lap_times'],
        #     'lap_counts': obs['lap_counts']
        # }

        # render object at each step
        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts'],
            'target_position_x': self.target_position[0][0],
            'target_position_y': self.target_position[0][1]
        }

        # times
        # reward function
        # print('3. target goal:', self.target_goal)
        # achieved_position = np.array([obs['poses_x'][0], obs['poses_y'][0], ])
        # achieved goal
        self.achieved_goal = np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0],
                                  obs['linear_vels_y'][0]])
        # print('obs poses_x in step() function', obs['poses_x'][0])

        # target goal
        self.target_goal = np.array([self.target_position[0][0], self.target_position[0][1], 1.57, 0, 0])
        # print('poses_x in step function:', obs['poses_x'])

        # reward = self.timestep
        # call reward function
        obstacle_danger, num_obstacle, avg_distance, min_distance = lidar_analysis(obs, self.bubble_radius)
        # reward = self.compute_reward(action, last_state, reset_flag,
        #                              [obstacle_danger, num_obstacle, avg_distance, min_distance],
        #                              list(obs['scans'][0]))

        # if episode_num == 0:  # reset at the start for every episode
        #     reward = 0
        # else:
        #     reward = self.compute_reward(episode_num, action, last_state, reset_flag,
        #                                  [obstacle_danger, num_obstacle, avg_distance, min_distance],
        #                                  list(obs['scans'][0]))

        reward = self.compute_reward(action, last_state, reset_flag,
                                    [obstacle_danger, num_obstacle, avg_distance, min_distance],
                                    list(obs['scans'][0]))


        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # check done
        # done, toggle_list = self._check_done()
        done = self._check_done()
        # info = {'checkpoint_done': toggle_list}

        # if self.error_turn_num >= 10:
        #     done = True

        info = {'check done': done}

        return obs, reward, done, info

    # def compute_reward(self, episode_num, action, last_state, reset_flag, scan_info_list, scans):
    def compute_reward(self, action, last_state, reset_flag, scan_info_list, scans):

        def compute_beta(delta_y, delta_x):
            beta = math.atan2(delta_y, delta_x)
            if delta_y < 0:
                beta = beta + 2 * np.pi

            return beta

        def compute_arc_diff(alpha, beta):
            if 0 <= alpha < np.pi:
                if alpha <= beta <= alpha + np.pi:
                    desired_steering_direction = 'left'
                    arc_diff = beta - alpha  # >=0
                elif 0 <= beta < alpha:
                    desired_steering_direction = 'right'
                    arc_diff = beta - alpha  # <0
                else:  # alpha + np.pi < beta < 2*np.pi
                    desired_steering_direction = 'right'
                    beta_main = beta - 2 * np.pi
                    arc_diff = beta_main - alpha  # <0
            elif alpha == np.pi:
                if 0 <= beta <= alpha:
                    desired_steering_direction = 'right'
                    arc_diff = beta - alpha  # <=0
                else:  # alpha < beta < 2*np.pi
                    desired_steering_direction = 'left'
                    arc_diff = beta - alpha  # >0
            else:  # np.pi < alpha < 2*np.pi
                if alpha - np.pi <= beta <= alpha:
                    desired_steering_direction = 'right'
                    arc_diff = beta - alpha  # <=0
                elif alpha < beta < 2 * np.pi:
                    desired_steering_direction = 'left'
                    arc_diff = beta - alpha  # >0
                else:  # 0 <= beta < alpha - np.pi
                    desired_steering_direction = 'left'
                    alpha_main = alpha - 2 * np.pi  # >0
                    arc_diff = beta - alpha_main

            return arc_diff


        # print('4. achieved goal:', self.achieved_goal)
        p = 0.5
        reach_cond = False
        position_achieved = np.array([self.achieved_goal[0], self.achieved_goal[1]])
        position_goal = np.array(self.target_position[0])
        position_cond = np.abs(position_achieved - position_goal)

        velocity_achieved = np.array([self.achieved_goal[3]])
        velocity_goal = np.array([self.target_goal[3]])
        velocity_cond = np.abs(velocity_achieved - velocity_goal)

        '''
        distance reward (s_i)
        '''
        pos_reward = -np.power(np.dot(np.abs(self.achieved_goal - self.target_goal), np.array(self.config["reward_weights"])), p)
        rate_reward = 0
        angle_reward = 0
        turn_correct = True

        if reset_flag:
            step_reward = pos_reward
            print('5.step reward:', step_reward)
        else:
            '''
            distance change rate reward (s_i-1, s_i)
            '''
            # last_pos = np.array([last_state[0], last_state[1]])
            last_pos = np.array([-last_state[0] + position_goal[0], -last_state[1] + position_goal[1]])
            last_distance = np.sqrt(np.sum(np.square(position_goal-last_pos)))
            current_distance = np.sqrt(np.sum(np.square(position_goal - position_achieved)))
            sensitivity_ratio = 10
            rate_weight = 10
            rate_reward = -math.tanh((current_distance - last_distance) * sensitivity_ratio) * rate_weight

            '''
            angle change reward  (s_i-1, a_i-1, s_i)
            '''
            # last step:
            # get heading angle (alpha) in the last step
            last_alpha = last_state[2]
            # get angle (beta) between target and achieved position in the last step
            # last_delta_y = position_goal[1] - last_state[1]
            # last_delta_x = position_goal[0] - last_state[0]
            last_delta_y = last_state[1]
            last_delta_x = last_state[0]
            last_beta = compute_beta(last_delta_y, last_delta_x)
            assert 0 <= last_alpha < 2 * np.pi and 0 <= last_beta < 2 * np.pi
            # get arc difference between beta and alpha in the last step
            last_arc_diff = compute_arc_diff(last_alpha, last_beta)

            # current step:
            # get heading angle (alpha) in current step
            alpha = self.achieved_goal[2]
            # get angle (beta) between target and achieved position in current step
            delta_y = position_goal[1] - position_achieved[1]
            delta_x = position_goal[0] - position_achieved[0]
            beta = compute_beta(delta_y, delta_x)
            assert 0 <= alpha < 2 * np.pi and 0 <= beta < 2 * np.pi
            # get arc difference between beta and alpha in current step
            arc_diff = compute_arc_diff(alpha, beta)

            # change arc to angle
            last_arc_diff_degree = last_arc_diff / np.pi * 180
            arc_diff_degree = arc_diff / np.pi * 180

            angle_weight = 1
            if last_arc_diff * action[0][0] < 0:
                angle_reward = -10
                self.error_turn_num = self.error_turn_num + 1
                turn_correct = False
            else:
                angle_reward = (abs(last_arc_diff_degree) - abs(arc_diff_degree)) * angle_weight

            # consider obstacles
            goal_radius = 1.5
            goal_reach_ratio = 2
            obstacle_danger, num_obstacle, avg_distance, min_distance = \
                scan_info_list[0], scan_info_list[1], scan_info_list[2], scan_info_list[3]

            # if episode_num <= 600:
            #     if not obstacle_danger:
            #         if episode_num <= 100:
            #             step_reward = (pos_reward + rate_reward + angle_reward) * goal_reach_ratio  # phase 1
            #         else:
            #             step_reward = (pos_reward * 2 + rate_reward * 2 + angle_reward) * goal_reach_ratio  # phase >=2
            #         if turn_correct:
            #             situation = 1
            #         else:
            #             situation = 2
            #         self.last_avg_distance = avg_distance
            #         return [step_reward, pos_reward, rate_reward, angle_reward, situation, turn_correct]
            #     else:
            #         situation = 3
            #         # step_reward_1 = pos_reward + rate_reward + angle_reward
            #         collide_reward_dict = self.compute_collide_reward(num_obstacle,
            #                                                           avg_distance, min_distance,
            #                                                           self.last_avg_distance)
            #         step_reward_2 = collide_reward_dict['collide_reward_1_2']
            #         step_reward = step_reward_2
            #         self.last_avg_distance = avg_distance
            #         return [step_reward, pos_reward, rate_reward, angle_reward, situation, collide_reward_dict]
            # else:
            #     if not obstacle_danger or current_distance < goal_radius:  # phase 3
            #         step_reward = (pos_reward * 2 + rate_reward * 2 + angle_reward) * goal_reach_ratio  # phase >=2
            #         if turn_correct:
            #             situation = 1
            #         else:
            #             situation = 2
            #         self.last_avg_distance = avg_distance
            #         return [step_reward, pos_reward, rate_reward, angle_reward, situation, turn_correct]
            #     else:
            #         situation = 3
            #         # step_reward_1 = pos_reward + rate_reward + angle_reward
            #         collide_reward_dict = self.compute_collide_reward(num_obstacle,
            #                                                           avg_distance, min_distance,
            #                                                           self.last_avg_distance)
            #         step_reward_2 = collide_reward_dict['collide_reward_3']
            #         step_reward = step_reward_2
            #         self.last_avg_distance = avg_distance
            #         return [step_reward, pos_reward, rate_reward, angle_reward, situation, collide_reward_dict]
            #


            if not obstacle_danger or current_distance < goal_radius:  # phase 3
            #  phase 1, 2
            # if not obstacle_danger:
            #     step_reward = (pos_reward + rate_reward + angle_reward) * goal_reach_ratio  # phase 1
                step_reward = (pos_reward * 2 + rate_reward * 2 + angle_reward) * goal_reach_ratio  # phase >=2
                if turn_correct:
                    situation = 1
                else:
                    situation = 2
                self.last_avg_distance = avg_distance
                return [step_reward, pos_reward, rate_reward, angle_reward, situation, turn_correct]
            else:
                situation = 3
                # step_reward_1 = pos_reward + rate_reward + angle_reward
                collide_reward_dict = self.compute_collide_reward(num_obstacle,
                                                                  avg_distance, min_distance, self.last_avg_distance)
                step_reward_2 = collide_reward_dict['collide_reward']
                step_reward = step_reward_2
                self.last_avg_distance = avg_distance
                return [step_reward, pos_reward, rate_reward, angle_reward, situation, collide_reward_dict]

    def compute_collide_reward(self, num_obstacle, avg_distance, min_distance, last_avg_distance):
        def get_collide_distance_reward(avg_distance):
            d_reward = -40 / (0.05 * avg_distance + 0.025) + 400
            return d_reward

        def get_collide_rate_reward(avg_distance, last_avg_distance):
            positive_sensitivity_ratio = 5
            negative_sensativiry_ratio = 5
            positive_rate_weight = 200
            negative_rate_weight = 200

            diff = avg_distance - last_avg_distance

            if diff >= 0:
                r_reward = math.tanh(diff * positive_sensitivity_ratio) * positive_rate_weight
            else:
                r_reward = math.tanh(diff * negative_sensativiry_ratio) * negative_rate_weight

            return r_reward

        distance_reward = get_collide_distance_reward(avg_distance)
        rate_reward = get_collide_rate_reward(avg_distance, last_avg_distance)

        collide_reward_dict = {
            'num_obstacle': num_obstacle,
            'avg_distance': avg_distance,
            'min_distance': min_distance,
            'last_avg_distance': last_avg_distance,
            'distance_reward': distance_reward,
            'rate_reward': rate_reward,
            'collide_reward': (distance_reward + rate_reward) * 0.1  # phase 1, 2
            # 'collide_reward': (distance_reward + rate_reward) * 0.5  # phase >=3
        }

        return collide_reward_dict

    # def reset(self, poses, goal_pos):
    def reset(self, poses, goal_pos):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # freespace
        # x, y = random.uniform(3.5, 25.5), random.uniform(3.5, 15)
        # self.target_position = [[x, y]]

        # obstacles
        # point = random.randint(1, 20)
        # if point == 1:
        #     x, y = random.uniform(14, 20), random.uniform(1.5, 2.5)
        #     self.target_position = [[x, y]]
        # elif point == 2:
        #     x, y = random.uniform(15.5, 18), random.uniform(7.2, 9)
        #     self.target_position = [[x, y]]
        # elif point == 3:
        #     x, y = random.uniform(13, 20), random.uniform(13, 15)
        #     self.target_position = [[x, y]]
        # elif point == 4:
        #     x, y = random.uniform(20, 22), random.uniform(15, 16)
        #     self.target_position = [[x, y]]
        # elif point == 5:
        #     x, y = random.uniform(22.2, 24), random.uniform(9, 12)
        #     self.target_position = [[x, y]]
        # elif point == 6:
        #     x, y = random.uniform(22.2, 25), random.uniform(3, 5)
        #     self.target_position = [[x, y]]
        # elif point == 7:
        #     x, y = random.uniform(1, 3), random.uniform(13, 15)
        #     self.target_position = [[x, y]]
        # elif point == 8:
        #     x, y = random.uniform(5.6, 7), random.uniform(7, 19)
        #     self.target_position = [[x, y]]
        # elif point == 9:
        #     x, y = random.uniform(24, 27), random.uniform(1, 16)
        #     self.target_position = [[x, y]]
        # else:
        #     x, y = random.uniform(3.5, 25.5), random.uniform(3.5, 15)
        #     self.target_position = [[x, y]]

        self.target_position = [goal_pos]
        self.target_goal = np.array(
            [self.target_position[0][0], self.target_position[0][1], 1.57, 0, 0])  # [x, y, heading angle, vx, vy]
        # print('target position in reset():', self.target_position)
        # print('target goal tuple in reset():', self.target_goal)
        # print('poses:', type(poses))

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])],
             [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))


        obs, reward, done, info = self.step(action)
        # obs, reward, done, info = self.step(0, action)


        # add target position [x,y]
        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts'],
            'target_position_x': self.target_position[0][0],
            'target_position_y': self.target_position[0][1]
        }
        # print('rendering observations at reset() function', self.render_obs)

        print('target place (as a member of input to the neural network', self.target_position)
        self.error_turn_num = 0
        return obs, reward, done, info, self.target_position[0]

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']

        if F110Env.renderer is None:
            # first call, initialize everything
            # from f110_gym.envs.rendering import EnvRenderer
            from multi_agent_sim.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            # F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H, self.render_obs['target_position_x'], self.render_obs['target_position_y'])
            F110Env.renderer.update_map(self.map_name, self.map_ext)
        # print('render observations in render function:', self.render_obs)
        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass