from multi_agent_sim.env_sac_f110 import F110Env
# from animation_plot import Animator
# import matplotlib.pyplot as plt
import yaml
import argparse
from argparse import Namespace
import numpy as np
import torch
import random
from collections import deque
import math

from multi_agent_sim.sac_naked import *
from multi_agent_sim.initialize import initialize_episode
from parl.utils import logger, summary

import os
import sys
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/multi_agent_sim/multi_agent_sim")

# import hyperparams
WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
FRAMES = 3


def maneuver(action):
    action[0] = action[0] * np.pi / 8  # steering angle
    action[1] = abs(action[1] * 4)  # velocity
    if action[0] < -np.pi / 8:
        action[0] = -np.pi / 8
    if action[0] > np.pi / 8:
        action[0] = np.pi / 8
    if action[1] > 4:
        action[1] = 4
    if action[1] < 0.8:
        action[1] = 0.8
    return action


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=999, type=int, help='Sets Gym seed')

    parser.add_argument("--save_root_path", default='./src/multi_agent_sim/multi_agent_sim/results/sac/train/', type=str, help='Sets Gym seed')
    parser.add_argument("--obs_dim", default=5, type=int, help='Manually set')
    parser.add_argument("--action_dim", default=2, type=int, help='Manually set')
    parser.add_argument(
        "--train_total_episodes",
        default=2500,
        type=int,
        help='Max number of episodes to run environment')
    parser.add_argument(
        "--max_episode_steps",
        default=12e2,
        type=int,
        help='Max number of steps to run environment')
    parser.add_argument(
        '--test_episode_interval',
        type=int,
        default=int(10),
        help='The number of episode interval between two consecutive evaluations')
    parser.add_argument(
        '--test_episode_number',
        type=int,
        default=int(5),
        help='The number of episodes in evaluation')
    parser.add_argument(
        "--alpha",
        default=0.2,
        type=float,
        help=
        'Determines the relative importance of entropy term against the reward'
    )
    args = parser.parse_args()

    logger.info("-------------------SAC TRAIN ---------------------")
    logger.info("--------------------------------------------------")


    # load map
    with open(path + '/src/multi_agent_sim/multi_agent_sim/new_map/map_sparse_obstacles.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # create env
    env = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=1, seed=args.seed)

    # Initialize model, algorithm, agent, replay_memory
    model = ACModel(FRAMES, args.obs_dim, args.action_dim)
    algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=args.alpha, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = SACAgent(algorithm)
    # # situation 1
    # # ratio_1 = 0.4   # phase 1
    # # ratio_1 = 0.8   # phase 2
    # ratio_1 = 0.7     # phase 3
    ratio_1 = 0.45   # TL
    rpm1 = ReplayMemory1(
        max_size=MEMORY_SIZE, obs_dim=args.obs_dim, laser_dim=1080, frames=FRAMES, act_dim=args.action_dim)
    # # situation 2
    # # ratio_2 = 0.4   # phase 1
    # # ratio_2 = 0.1   # phase 2
    # ratio_2 = 0.1     # phase 3
    ratio_2 = 0.2   # TL
    rpm2 = ReplayMemory2(
        max_size=MEMORY_SIZE, obs_dim=args.obs_dim, laser_dim=1080, frames=FRAMES, act_dim=args.action_dim)
    # # situation 3
    # # ratio_3 = 0.2   # phase 1
    # # ratio_3 = 0.1   # phase 2
    # ratio_3 = 0.2     # phase 3
    ratio_3 = 0.35  # TL

    rpm3 = ReplayMemory3(
        max_size=MEMORY_SIZE, obs_dim=args.obs_dim, laser_dim=1080, frames=FRAMES, act_dim=args.action_dim)

    # # load model
    # last_model_path = args.save_root_path + str(args.seed) + '/model/sac_last.ckpt'
    # last_model_path = args.save_root_path + str(args.seed) + '/model/sac_last.ckpt'
    # agent.restore(last_model_path)

    total_step = 0
    episode_reward_list = [0]
    evaluate_reward_list = [0]
    best_episode_reward = -9999999999.9
    for episode_num in range(args.train_total_episodes):

        print('episode:', episode_num + 1)
        scans_stack = deque(maxlen=FRAMES)

        sample_3_num = math.ceil(BATCH_SIZE * ratio_3)
        sample_2_num = math.ceil(BATCH_SIZE * ratio_2)
        sample_1_num = BATCH_SIZE - sample_3_num - sample_2_num

        # initialization
        start_pos_theta, goal_pos = initialize_episode()

        # obs, step_reward, done, info, position_goal = env.reset(np.array([start_pos_theta]), goal_pos)
        obs, step_reward, done, info, position_goal = env.reset(np.array([start_pos_theta]), goal_pos)

        target_x = position_goal[0]
        target_y = position_goal[1]
        # env.render()

        scans_stack.append(obs['scans'][0])
        scans_stack.append(obs['scans'][0])
        scans_stack.append(obs['scans'][0])

        scans = copy.deepcopy(scans_stack)
        state = np.array(
            [target_x - obs['poses_x'][0], target_y - obs['poses_y'][0],
             obs['poses_theta'][0],
             obs['linear_vels_x'][0],
             obs['ang_vels_z'][0]])

        situation = 1
        collide = False
        episode_reward = 0
        step = 0
        while not done and step < args.max_episode_steps:
            # print('simtime', simtime)
            print('--------------------------------------------')
            step = step + 1
            total_step = total_step + 1
            print('Episode: {}, Step: {}, Total step: {}'.format(episode_num + 1, step, total_step))
            while True:
                action = agent.sample(scans, FRAMES, state)
                action = maneuver(action)

                # step function
                obs_prime, step_reward, done, info = env.step(np.array([action]), state, reset_flag=False)
                # obs_prime, step_reward, done, info = env.step(num_episode, np.array([action]), state, reset_flag=False)

                enough_sample = rpm1.size() > sample_1_num and rpm2.size() > sample_2_num and \
                                rpm3.size() > sample_3_num and (rpm1.size() + rpm2.size() + rpm3.size() > WARMUP_STEPS)

                if not enough_sample:
                    break
                else:
                    situation = step_reward[-2]
                    if situation == 1:  # not danger, turn correct
                        break
                    if situation == 2:  # not danger, turn wrong
                        point = random.randint(1, 100)
                        # if point <= 20:  # phase 1, 2
                        # if point <= 80:    # phase >= 2
                        if point <= 50:  # TL
                            break
                    if situation == 3:  # close to danger
                        min_scan = step_reward[-1]['min_distance']
                        if min_scan >= 0.4:
                            break
                        else:
                            point = random.randint(1, 100)
                            # if point <= 40:   # phase 1, 2
                            if point <= 80:     # phase 3, TL
                                break

            print('1. state:', state)
            print('2. action:', action)
            print('3. step reward:', step_reward)
            print('4. info:', info)
            print('5. obs_prime:', obs_prime)

            scans_stack.append(obs_prime['scans'][0])
            state_prime = np.array([target_x - obs_prime['poses_x'][0],  target_y - obs_prime['poses_y'][0],
                                    obs_prime['poses_theta'][0],
                                    obs_prime['linear_vels_x'][0],
                                    obs_prime['ang_vels_z'][0]])

            print('6. state_prime:', state_prime)
            print('situation:', step_reward[-2])
            situation = step_reward[-2]
            if situation == 1:
                rpm1.append(state, scans, action, step_reward[0], state_prime, scans_stack, done)
            if situation == 2:
                rpm2.append(state, scans, action, step_reward[0], state_prime, scans_stack, done)
            if situation == 3:
                rpm3.append(state, scans, action, step_reward[0], state_prime, scans_stack, done)
            scans = copy.deepcopy(scans_stack)
            state = state_prime

            # train
            print(rpm1.size(), rpm2.size(), rpm3.size())
            if rpm1.size() > sample_1_num and rpm2.size() > sample_2_num and rpm3.size() > sample_3_num and \
                    (rpm1.size() + rpm2.size() + rpm3.size() > WARMUP_STEPS):
                print('-- train start --')
                batch_obs_3, batch_scans_3, batch_action_3, batch_reward_3, batch_next_obs_3, batch_next_scans_3, batch_terminal_3 \
                    = rpm3.sample_batch(sample_3_num)
                batch_obs_2, batch_scans_2, batch_action_2, batch_reward_2, batch_next_obs_2, batch_next_scans_2, batch_terminal_2 \
                    = rpm2.sample_batch(sample_2_num)
                batch_obs_1, batch_scans_1, batch_action_1, batch_reward_1, batch_next_obs_1, batch_next_scans_1, batch_terminal_1 \
                    = rpm1.sample_batch(sample_1_num)

                batch_obs = np.vstack((batch_obs_3, batch_obs_2, batch_obs_1))
                batch_scans = np.vstack((batch_scans_3, batch_scans_2, batch_scans_1))
                batch_action = np.vstack((batch_action_3, batch_action_2, batch_action_1))
                batch_reward = np.concatenate((batch_reward_3, batch_reward_2, batch_reward_1), axis=0)
                batch_next_obs = np.vstack((batch_next_obs_3, batch_next_obs_2, batch_next_obs_1))
                batch_next_scans = np.vstack((batch_next_scans_3, batch_next_scans_2, batch_next_scans_1))
                batch_terminal = np.concatenate((batch_terminal_3, batch_terminal_2, batch_terminal_1), axis=0)

                permutation = np.random.permutation(batch_obs.shape[0])
                batch_obs = batch_obs[permutation, :]
                batch_scans = batch_scans[permutation, :, :]
                batch_action = batch_action[permutation, :]
                batch_reward = batch_reward[permutation]
                batch_next_obs = batch_next_obs[permutation, :]
                batch_next_scans = batch_next_scans[permutation, :, :]
                batch_terminal = batch_terminal[permutation]

                agent.learn(batch_obs, batch_scans, batch_action, batch_reward,
                            batch_next_obs, batch_next_scans, batch_terminal)
                print('-- train end -- ')

            env.render(mode='human')
            print('7. done info:', done)

            if not done:
                # episode reward
                episode_reward += step_reward[0]
                print('8. episode reward:', episode_reward)

        # write to logger
        summary.add_scalar('train/episode_reward', episode_reward, total_step)
        logger.info('Episode: {} Total Steps: {} Reward: {}'.format(
            (episode_num + 1), total_step, episode_reward))

        # evaluate episode
        if (episode_num + 1) % args.test_episode_interval == 0:

            avg_reward = run_evaluate_episodes(agent, env, args.test_episode_number, conf)
            # avg_reward = run_evaluate_episodes(episode_num+1, agent, env, args.test_episode_number, conf)

            evaluate_reward_list.append(avg_reward)
            summary.add_scalar('eval/episode_reward', avg_reward, total_step)
            logger.info('Evaluation over: {} episodes, Average episode reward: {}'.format(
                args.test_episode_number, avg_reward))

        # save model
        last_model_path = args.save_root_path + str(args.seed) + '/model/sac_last.ckpt'
        agent.save(last_model_path)
        if episode_reward > best_episode_reward:
            best_episode_index = episode_num + 1
            best_episode_reward = episode_reward
            best_model_path = args.save_root_path + str(args.seed) + '/model/sac_best.ckpt'
            agent.save(best_model_path)

        # save info
        print('The total reward for episode', episode_num + 1, ' is:', episode_reward)

        episode_reward_list.append(episode_reward)
        info_dict = {
            'episode_reward_list': episode_reward_list,
            'evaluate_reward_list': evaluate_reward_list,
            'best_episode_reward': best_episode_reward,
            'best_episode_index': best_episode_index
        }
        torch.save(info_dict, args.save_root_path + str(args.seed) + '/info_dict')

    env.close()


# plt.show()


# def run_evaluate_episodes(episode_num, agent, env, eval_episodes, conf):
def run_evaluate_episodes(agent, env, eval_episodes, conf):
    total_reward = 0
    for _ in range(eval_episodes):
        scans_stack = deque(maxlen=FRAMES)
        start_pos_theta, goal_pos = initialize_episode()
        obs, step_reward, done, info, position_goal = env.reset(np.array([start_pos_theta]), goal_pos)
        target_x = position_goal[0]
        target_y = position_goal[1]

        scans_stack.append(obs['scans'][0])
        scans_stack.append(obs['scans'][0])
        scans_stack.append(obs['scans'][0])

        scans = copy.deepcopy(scans_stack)
        state = np.array(
            [target_x - obs['poses_x'][0], target_y - obs['poses_y'][0],
             obs['poses_theta'][0],
             obs['linear_vels_x'][0],
             obs['ang_vels_z'][0]])

        step = 0
        while not done and step < args.max_episode_steps:
            step = step + 1
            action = agent.predict(scans, FRAMES, state)
            action = maneuver(action)

            obs_prime, step_reward, done, info = env.step(np.array([action]), state, reset_flag=False)
            # obs_prime, step_reward, done, info = env.step(episode_num, np.array([action]), state, reset_flag=False)


            scans_stack.append(obs_prime['scans'][0])
            state_prime = np.array([target_x - obs_prime['poses_x'][0],  target_y - obs_prime['poses_y'][0],
                                    obs_prime['poses_theta'][0],
                                    obs_prime['linear_vels_x'][0],
                                    obs_prime['ang_vels_z'][0]])

            scans = copy.deepcopy(scans_stack)
            state = state_prime
            if not done:
                total_reward += step_reward[0]

    avg_reward = total_reward / eval_episodes
    return avg_reward


if __name__ == '__main__':


    main()