# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from mixed_reality_py.base_classes import Simulator
from mixed_reality_py.lidar_process import lidar_analysis
# others
import numpy as np
import os
import time
import random
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
            # self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
            #                'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
            #                'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31,
            #                'length': 0.58}

            # params from 5 traces per round for No.10 training result (15 rounds)
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.16816832, 'lr': 0.15177377, 'h': 0.074,
                           'm': 3.74, 'I': 0.04712, 's_min': -0.52231318, 's_max': 0.51628776, 'sv_min': -3.42537148, 'sv_max': 3.88872181,
                           'v_switch': 8.53771819, 'a_max': 9.6069614, 'v_min': -4.94960142, 'v_max': 19.96697248, 'width': 0.31,
                           'length': 0.58}


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

        # # loop completion
        # self.near_start = True
        # self.num_toggles = 0
        #
        # # race info
        # self.lap_times = np.zeros((self.num_agents,))
        # self.lap_counts = np.zeros((self.num_agents,))
        # self.current_time = 0.0
        #
        # # finish line info
        # self.num_toggles = 0
        # self.near_start = True
        # self.near_starts = np.array([True] * self.num_agents)
        # self.toggle_list = np.zeros((self.num_agents,))
        # self.start_xs = np.zeros((self.num_agents,))
        # self.start_ys = np.zeros((self.num_agents,))
        # self.start_thetas = np.zeros((self.num_agents,))
        # self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

        self.target_position = []
        self.target_goal = []
        self.config = {"reward_weights": [1, 1, 0, 0, 0]}
        self.achieved_goal = []
        self.bubble_radius = 1.5
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

        # position goal
        # reach_cond = False
        # position_achieved = np.array([self.achieved_goal[0], self.achieved_goal[1]])
        # position_goal = np.array(self.target_position[0])
        # position_cond = np.abs(position_achieved - position_goal)
        # # print('position reach condition:', position_cond)

        # velocity_achieved = np.array([self.achieved_goal[3]])
        # velocity_goal = np.array([self.target_goal[3]])
        # velocity_cond = np.abs(velocity_achieved - velocity_goal)
        # # print('velocity reach condition:', velocity_cond)
        #
        # if position_cond[0] < 0.4 and position_cond[1] < 0.4:
        #     # if velocity_cond.any() == 0:
        #     reach_cond = True
        #         # print('reach condition:', reach_cond)
        #
        # done = (self.collisions[self.ego_idx]) or reach_cond

        # position goal
        reach_cond_1 = False
        position_achieved_1 = np.array([self.achieved_goals[0][0], self.achieved_goals[0][1]])
        position_goal_1 = np.array([self.target_positions[0][0], self.target_positions[0][1]])
        position_cond_1 = np.abs(position_achieved_1 - position_goal_1)

        reach_cond_2 = False
        position_achieved_2 = np.array([self.achieved_goals[1][0], self.achieved_goals[1][1]])
        position_goal_2 = np.array([self.target_positions[1][0], self.target_positions[1][1]])
        position_cond_2 = np.abs(position_achieved_2 - position_goal_2)

        # reach_cond_3 = False
        # position_achieved_3 = np.array([self.achieved_goals[2][0], self.achieved_goals[2][1]])
        # position_goal_3 = np.array([self.target_positions[2][0], self.target_positions[2][1]])
        # position_cond_3 = np.abs(position_achieved_3 - position_goal_3)

        # reach_cond_4 = False
        # position_achieved_4 = np.array([self.achieved_goals[3][0], self.achieved_goals[3][1]])
        # position_goal_4 = np.array([self.target_positions[3][0], self.target_positions[3][1]])
        # position_cond_4 = np.abs(position_achieved_4 - position_goal_4)

        # velocity goal
        velocity_achieved_1 = np.array([self.achieved_goals[0][3]])
        velocity_goal_1 = np.array([self.target_goals[0][3]])
        velocity_cond_1 = np.abs(velocity_achieved_1 - velocity_goal_1)

        velocity_achieved_2 = np.array([self.achieved_goals[1][3]])
        velocity_goal_2 = np.array([self.target_goals[1][3]])
        velocity_cond_2 = np.abs(velocity_achieved_2 - velocity_goal_2)

        # velocity_achieved_3 = np.array([self.achieved_goals[2][3]])
        # velocity_goal_3 = np.array([self.target_goals[2][3]])
        # velocity_cond_3 = np.abs(velocity_achieved_3 - velocity_goal_3)
        #
        # velocity_achieved_4 = np.array([self.achieved_goals[3][3]])
        # velocity_goal_4 = np.array([self.target_goals[3][3]])
        # velocity_cond_4 = np.abs(velocity_achieved_4 - velocity_goal_4)

        if position_cond_1[0] < 0.4 and position_cond_1[1] < 0.4:
            reach_cond_1 = True
        if position_cond_2[0] < 0.4 and position_cond_2[1] < 0.4:
            reach_cond_2 = True
        # if position_cond_3[0] < 0.4 and position_cond_3[1] < 0.4:
        #     reach_cond_3 = True
        # if position_cond_4[0] < 0.4 and position_cond_4[1] < 0.4:
        #     reach_cond_4 = True

        # done = (self.collisions[0] or self.collisions[1] or self.collisions[2] or self.collisions[3]) \
        #        or (reach_cond_1 and reach_cond_2 and reach_cond_3 and reach_cond_4)

        done = (self.collisions[0] or self.collisions[1]) \
                or (reach_cond_1 and reach_cond_2)

        done_info_dict = {
            'reach_cond_1': reach_cond_1,
            'reach_cond_2': reach_cond_2,
            # 'reach_cond_3': reach_cond_3,
            # 'reach_cond_4': reach_cond_4,
            'collision_1': self.collisions[0],
            'collision_2': self.collisions[1]
            # 'collision_3': self.collisions[2],
            # 'collision_4': self.collisions[3]
        }

        return done, done_info_dict

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

        # call simulation step
        # print('1. action:', action)
        print("last state is :", last_state)
        obs = self.sim.step(action)
        print("obs is :", obs)
        # to implement digital twin
        if len(last_state) > 0:                      
            new_obs = {}
            new_obs["ego_idx"] = 0
            new_obs["scans"] = obs['scans']
            new_obs["poses_x"] = [obs['poses_x'][0], last_state[1][0]]
            new_obs["poses_y"] = [obs['poses_y'][0], last_state[1][1]]
            new_obs["poses_theta"] = [obs['poses_theta'][0], last_state[1][2]]
            new_obs["linear_vels_x"] = [obs['linear_vels_x'][0], last_state[1][3]]
            new_obs["linear_vels_y"] = obs['linear_vels_y']
            new_obs["ang_vels_z"] = [obs['ang_vels_z'][0], last_state[1][4]]
            new_obs["collisions"] = obs['collisions']
            print("new_obs is:", new_obs)
        else:
            new_obs = obs


        # F110Env.current_obs = obs
        F110Env.current_obs = new_obs
        # print('3. obs:', obs)
        # print('----------------------')
        # render object at each step
        # self.render_obs = {
        #     'ego_idx': obs['ego_idx'],
        #     'poses_x': obs['poses_x'],
        #     'poses_y': obs['poses_y'],
        #     'poses_theta': obs['poses_theta'],
        #     # 'lap_times': obs['lap_times'],
        #     # 'lap_counts': obs['lap_counts'],
        #     'target_position': self.target_positions
        # }

        self.render_obs = {
            'ego_idx': new_obs['ego_idx'],
            'poses_x': new_obs['poses_x'],
            'poses_y': new_obs['poses_y'],
            'poses_theta': new_obs['poses_theta'],
            # 'lap_times': obs['lap_times'],
            # 'lap_counts': obs['lap_counts'],
            'target_position': self.target_positions
        }
        print("render_obs:", self.render_obs)
        # self.render(mode='human')

        # self.achieved_goals = np.array([
        #     [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
        #      obs['linear_vels_x'][0], obs['linear_vels_y'][0]],
        #     [obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],
        #      obs['linear_vels_x'][1], obs['linear_vels_y'][1]]
        # ])

        self.achieved_goals = np.array([
            [new_obs['poses_x'][0], new_obs['poses_y'][0], new_obs['poses_theta'][0],
             new_obs['linear_vels_x'][0], new_obs['linear_vels_y'][0]],
            [new_obs['poses_x'][1], new_obs['poses_y'][1], new_obs['poses_theta'][1],
             new_obs['linear_vels_x'][1], new_obs['linear_vels_y'][1]]
        ])

        # agent 1
        obstacle_danger, num_obstacle, avg_distance, min_distance = \
            lidar_analysis(new_obs, self.bubble_radius, agent_index=0)
        self.target_goal = self.target_goals[0]
        self.target_position = self.target_positions[0]
        self.achieved_goal = self.achieved_goals[0]
        if reset_flag:
            reward_1 = self.compute_reward([action[0]], [], reset_flag,
                                           [obstacle_danger, num_obstacle, avg_distance, min_distance],
                                           list(new_obs['scans'][0]))
        else:
            reward_1 = self.compute_reward([action[0]], last_state[0], reset_flag,
                                           [obstacle_danger, num_obstacle, avg_distance, min_distance],
                                           list(new_obs['scans'][0]))

        # agent 2
        obstacle_danger, num_obstacle, avg_distance, min_distance = \
            lidar_analysis(new_obs, self.bubble_radius, agent_index=1)
        self.target_goal = self.target_goals[1]
        self.target_position = self.target_positions[1]
        self.achieved_goal = self.achieved_goals[1]
        if reset_flag:
            reward_2 = self.compute_reward([action[1]], [], reset_flag,
                                           [obstacle_danger, num_obstacle, avg_distance, min_distance],
                                           list(new_obs['scans'][1]))
        else:
            reward_2 = self.compute_reward([action[1]], last_state[1], reset_flag,
                                           [obstacle_danger, num_obstacle, avg_distance, min_distance],
                                           list(new_obs['scans'][1]))

        # # agent 3
        # obstacle_danger, num_obstacle, avg_distance, min_distance = \
        #     lidar_analysis(obs, self.bubble_radius, agent_index=2)
        # self.target_goal = self.target_goals[2]
        # self.target_position = self.target_positions[2]
        # self.achieved_goal = self.achieved_goals[2]
        # if reset_flag:
        #     reward_3 = self.compute_reward([action[2]], [], reset_flag,
        #                                    [obstacle_danger, num_obstacle, avg_distance, min_distance],
        #                                    list(obs['scans'][2]))
        # else:
        #     reward_3 = self.compute_reward([action[2]], last_state[2], reset_flag,
        #                                    [obstacle_danger, num_obstacle, avg_distance, min_distance],
        #                                    list(obs['scans'][2]))
        #
        # # agent 4
        # obstacle_danger, num_obstacle, avg_distance, min_distance = \
        #     lidar_analysis(obs, self.bubble_radius, agent_index=3)
        # self.target_goal = self.target_goals[3]
        # self.target_position = self.target_positions[3]
        # self.achieved_goal = self.achieved_goals[3]
        # if reset_flag:
        #     reward_4 = self.compute_reward([action[3]], [], reset_flag,
        #                                    [obstacle_danger, num_obstacle, avg_distance, min_distance],
        #                                    list(obs['scans'][3]))
        # else:
        #     reward_4 = self.compute_reward([action[3]], last_state[3], reset_flag,
        #                                    [obstacle_danger, num_obstacle, avg_distance, min_distance],
        #                                    list(obs['scans'][3]))

        # update data member
        self._update_state(new_obs)
        # self._update_state(obs)

        # check done
        done, done_info_dict = self._check_done()
        info = done_info_dict

        # return obs, [reward_1, reward_2, reward_3, reward_4], done, info

        # return obs, [reward_1, reward_2], done, info
        return new_obs, [reward_1, reward_2], done, info

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
        position_goal = np.array([self.target_position[0], self.target_position[1]])
        position_cond = np.abs(position_achieved - position_goal)

        velocity_achieved = np.array([self.achieved_goal[3]])
        velocity_goal = np.array([self.target_goal[3]])
        velocity_cond = np.abs(velocity_achieved - velocity_goal)

        '''
        distance reward (s_i)
        '''
        pos_reward = -np.power(
            np.dot(np.abs(self.achieved_goal - self.target_goal), np.array(self.config["reward_weights"])), p)
        rate_reward = 0
        angle_reward = 0
        turn_correct = True

        if reset_flag:
            step_reward = pos_reward
            # print('5.step reward:', step_reward)
        else:
            '''
            distance change rate reward (s_i-1, s_i)
            '''
            # last_pos = np.array([last_state[0], last_state[1]])
            last_pos = np.array([-last_state[0] + position_goal[0], -last_state[1] + position_goal[1]])
            last_distance = np.sqrt(np.sum(np.square(position_goal - last_pos)))
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
            # assert 0 <= last_alpha < 2 * np.pi and 0 <= last_beta < 2 * np.pi
            # get arc difference between beta and alpha in the last step
            last_arc_diff = compute_arc_diff(last_alpha, last_beta)

            # current step:
            # get heading angle (alpha) in current step
            alpha = self.achieved_goal[2]
            # get angle (beta) between target and achieved position in current step
            delta_y = position_goal[1] - position_achieved[1]
            delta_x = position_goal[0] - position_achieved[0]
            beta = compute_beta(delta_y, delta_x)
            # assert 0 <= alpha < 2 * np.pi and 0 <= beta < 2 * np.pi
            # get arc difference between beta and alpha in current step
            arc_diff = compute_arc_diff(alpha, beta)

            # change arc to angle
            last_arc_diff_degree = last_arc_diff / np.pi * 180
            arc_diff_degree = arc_diff / np.pi * 180

            angle_weight = 1
            if last_arc_diff * action[0][0] < 0:
                angle_reward = -10
                turn_correct = False
            else:
                angle_reward = (abs(last_arc_diff_degree) - abs(arc_diff_degree)) * angle_weight

            # consider obstacles
            goal_radius = 1.5
            goal_reach_ratio = 2
            obstacle_danger, num_obstacle, avg_distance, min_distance = \
                scan_info_list[0], scan_info_list[1], scan_info_list[2], scan_info_list[3]

            if not obstacle_danger or current_distance < goal_radius:
                # if not obstacle_danger:
                # step_reward = (pos_reward + rate_reward + angle_reward) * goal_reach_ratio
                step_reward = (pos_reward * 2 + rate_reward * 2 + angle_reward) * goal_reach_ratio
                if current_distance < goal_radius:
                    step_reward = (pos_reward * 4 + rate_reward * 4 + angle_reward) * goal_reach_ratio
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
            # 'collide_reward': (distance_reward + rate_reward) * 0.1
            # 'collide_reward': (distance_reward + rate_reward) * 0.5,
            'collide_reward': (distance_reward + rate_reward) * 0.3
        }

        return collide_reward_dict

    def reset(self, initial_list):
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
        start_pos_1_x = initial_list[0][0]
        start_pos_1_y = initial_list[0][1]
        start_theta_1 = initial_list[0][2]
        goal_pos_1_x = initial_list[0][3]
        goal_pos_1_y = initial_list[0][4]

        start_pos_2_x = initial_list[1][0]
        start_pos_2_y = initial_list[1][1]
        start_theta_2 = initial_list[1][2]
        goal_pos_2_x = initial_list[1][3]
        goal_pos_2_y = initial_list[1][4]

        # start_pos_3_x = initial_list[2][0]
        # start_pos_3_y = initial_list[2][1]
        # start_theta_3 = initial_list[2][2]
        # goal_pos_3_x = initial_list[2][3]
        # goal_pos_3_y = initial_list[2][4]
        #
        # start_pos_4_x = initial_list[3][0]
        # start_pos_4_y = initial_list[3][1]
        # start_theta_4 = initial_list[3][2]
        # goal_pos_4_x = initial_list[3][3]
        # goal_pos_4_y = initial_list[3][4]

        # self.target_positions = np.array([[goal_pos_1_x, goal_pos_1_y, 0],  # the last 0 is theta of plotting targets
        #                                   [goal_pos_2_x, goal_pos_2_y, 0],
        #                                   [goal_pos_3_x, goal_pos_3_y, 0],
        #                                   [goal_pos_4_x, goal_pos_4_y, 0]])


        self.target_positions = np.array([[goal_pos_1_x, goal_pos_1_y, 0],  # the last 0 is theta of plotting targets
                                          [goal_pos_2_x, goal_pos_2_y, 0]
                                          ])

        # self.target_goals = np.array([[goal_pos_1_x, goal_pos_1_y, 1.57, 0, 0],
        #                               [goal_pos_2_x, goal_pos_2_y, 1.57, 0, 0],
        #                               [goal_pos_3_x, goal_pos_3_y, 1.57, 0, 0],
        #                               [goal_pos_4_x, goal_pos_4_y, 1.57, 0, 0]])  # [x, y, heading angle, vx, vy]

        self.target_goals = np.array([[goal_pos_1_x, goal_pos_1_y, 1.57, 0, 0],
                                      [goal_pos_2_x, goal_pos_2_y, 1.57, 0, 0]
                                      ])  # [x, y, heading angle, vx, vy]

        # call reset to simulator
        # poses = np.array([[start_pos_1_x, start_pos_1_y, start_theta_1],
        #                   [start_pos_2_x, start_pos_2_y, start_theta_2],
        #                   [start_pos_3_x, start_pos_3_y, start_theta_3],
        #                   [start_pos_4_x, start_pos_4_y, start_theta_4]])

        poses = np.array([[start_pos_1_x, start_pos_1_y, start_theta_1],
                          [start_pos_2_x, start_pos_2_y, start_theta_2]])

        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, rewards, done, info = self.step(action)

        # add target position [x,y]
        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            # 'lap_times': obs['lap_times'],
            # 'lap_counts': obs['lap_counts'],
            'target_position': self.target_positions
        }
        # print('rendering observations at reset() function', self.render_obs)

        return obs, rewards, done, info, self.target_positions


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
            from rendering import EnvRenderer
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
            # time.sleep(10)
        elif mode == 'human_fast':
            pass
