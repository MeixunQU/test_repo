import paddle
from paddle.nn import Conv1D, Linear
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Normal
# import gym
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from tqdm import tqdm
import numpy as np
import copy
import parl
# import torch

# clamp bounds for std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0

'''
Model
'''
class ACModel(paddle.nn.Layer):
    def __init__(self, frames, obs_dim, action_dim):
        super(ACModel, self).__init__()
        self.actor_model = Actor(frames, obs_dim, action_dim)
        self.critic_model = Critic(frames, obs_dim, action_dim)

    def policy(self, scans, obs):
        return self.actor_model(scans, obs)

    def value(self, scans, obs, action):
        return self.critic_model(scans, obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def sync_weights_to(self, target_model, decay=0.0):
        target_vars = dict(target_model.named_parameters())
        for name, var in self.named_parameters():
            target_data = decay * target_vars[name] + (1 - decay) * var
            target_vars[name] = target_data
        target_model.set_state_dict(target_vars)


# class Actor(paddle.nn.Layer):
#     def __init__(self, frames, obs_dim, action_dim):
#         super(Actor, self).__init__()
#
#         self.l1 = nn.Linear(obs_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.mean_linear = nn.Linear(256, action_dim)
#         self.std_linear = nn.Linear(256, action_dim)
#
#     def forward(self, obs):
#         x = F.relu(self.l1(obs))
#         x = F.relu(self.l2(x))
#
#         act_mean = self.mean_linear(x)
#         act_std = self.std_linear(x)
#         act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return act_mean, act_log_std

class Actor(parl.Model):
    def __init__(self, frames, obs_dim, action_dim):
        super(Actor, self).__init__()
        # I. cnn module
        self.conv1 = Conv1D(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = Conv1D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = Linear(in_features=8640, out_features=256)

        # II. feedforward module
        self.l1 = nn.Linear(256+obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)
        self.std_linear = nn.Linear(256, action_dim)

    def forward(self, scans, obs):
        # I. cnn module
        x = F.relu(self.conv1(scans))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = paddle.reshape(x, [x.shape[0], -1])
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        combine_obs = paddle.concat(x=[x, obs], axis=-1)

        # II. feedforward module
        y = F.relu(self.l1(combine_obs))
        y = F.relu(self.l2(y))
        act_mean = self.mean_linear(y)
        act_std = self.std_linear(y)
        act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # print('act_mean:', act_mean)
        # print('act_log_std:', act_log_std)
        return act_mean, act_log_std


# class Critic(paddle.nn.Layer):
#     def __init__(self, frames, obs_dim, action_dim):
#         super(Critic, self).__init__()
#
#         # Q1 network
#         self.l1 = nn.Linear(obs_dim + action_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 1)
#
#         # Q2 network
#         self.l4 = nn.Linear(obs_dim + action_dim, 256)
#         self.l5 = nn.Linear(256, 256)
#         self.l6 = nn.Linear(256, 1)
#
#     def forward(self, obs, action):
#         x = paddle.concat([obs, action], 1)
#
#         # Q1
#         q1 = F.relu(self.l1(x))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)
#
#         # Q2
#         q2 = F.relu(self.l4(x))
#         q2 = F.relu(self.l5(q2))
#         q2 = self.l6(q2)
#         return q1, q2

class Critic(parl.Model):
    def __init__(self, frames, obs_dim, action_dim):
        super(Critic, self).__init__()
        # I. cnn module
        self.conv1 = Conv1D(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = Conv1D(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = Linear(in_features=8640, out_features=256)

        # Q1 network
        self.l1 = nn.Linear(256 + obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 network
        self.l4 = nn.Linear(256 + obs_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, scans, obs, action):
        # I. cnn module
        x = F.relu(self.conv1(scans))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = paddle.reshape(x, [x.shape[0], -1])
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)

        combine_input = paddle.concat(x=[x, obs, action], axis=-1)

        # Q1
        q1 = F.relu(self.l1(combine_input))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        q2 = F.relu(self.l4(combine_input))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


'''
Algorithm
'''
class SAC():
    def __init__(self, model, gamma=None, tau=None, alpha=None, actor_lr=None, critic_lr=None):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, scans, obs):
        act_mean, _ = self.model.policy(scans, obs)
        action = paddle.tanh(act_mean)
        return action

    def sample(self, scans, obs):
        act_mean, act_log_std = self.model.policy(scans, obs)
        normal = Normal(act_mean, 1 * act_log_std.exp())
        # 重参数化  (mean + std*N(0,1))
        x_t = normal.sample([1])
        action = paddle.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
        log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)
        return action[0], log_prob[0]

    def save(self):
        paddle.save(self.model.actor_model.state_dict(), 'net.pdparams')

    def learn(self, obs, scans, action, reward, next_obs, next_scans, terminal):
        critic_loss = self._critic_learn(scans, obs, action, reward, next_scans, next_obs, terminal)
        actor_loss = self._actor_learn(scans, obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, scans, obs, action, reward, next_scans, next_obs, terminal):
        with paddle.no_grad():
            next_action, next_log_pro = self.sample(next_scans, next_obs)
            q1_next, q2_next = self.target_model.value(next_scans, next_obs, next_action)
            target_Q = paddle.minimum(q1_next,
                                      q2_next) - self.alpha * next_log_pro
            terminal = paddle.cast(terminal, dtype='float32')
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        cur_q1, cur_q2 = self.model.value(scans, obs, action)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, scans, obs):
        act, log_pi = self.sample(scans, obs)
        q1_pi, q2_pi = self.model.value(scans, obs, act)
        min_q_pi = paddle.minimum(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)


'''
Agent
'''
class SACAgent(parl.Agent):
    def __init__(self, algorithm):
        self.alg = algorithm
        self.alg.sync_target(decay=0)

    def predict(self, scans, frames, obs):
        scans = np.array(scans)
        scans = paddle.to_tensor(scans.reshape(1, frames, -1), dtype='float32')
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        # print('obs_tensor:\n', obs)
        action = self.alg.predict(scans, obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def sample(self, scans, frames, obs):
        scans = np.array(scans)
        scans = paddle.to_tensor(scans.reshape(1, frames, -1), dtype='float32')
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action, _ = self.alg.sample(scans, obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, scans, action, reward, next_obs, next_scans, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        scans = paddle.to_tensor(scans, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        next_scans = paddle.to_tensor(next_scans, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, scans, action, reward, next_obs, next_scans, terminal)

        return critic_loss, actor_loss


'''
Buffer
'''
class ReplayMemory1(object):
    def __init__(self, max_size, obs_dim, laser_dim, frames, act_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.laser_dim = laser_dim
        self.act_dim = act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.scans = np.zeros((max_size, frames, laser_dim), dtype='float32')
        self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size,), dtype='float32')
        self.terminal = np.zeros((max_size,), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_scans = np.zeros((max_size, frames, laser_dim), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    # 抽样指定数量（batch_size）的经验
    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        obs = self.obs[batch_idx]
        scans = self.scans[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        next_scans = self.next_scans[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, scans, action, reward, next_obs, next_scans, terminal

    def append(self, obs, scans, act, reward, next_obs, next_scans, terminal):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.scans[self._curr_pos] = scans
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.next_scans[self._curr_pos] = next_scans
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size

class ReplayMemory2(object):
    def __init__(self, max_size, obs_dim, laser_dim, frames, act_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.laser_dim = laser_dim
        self.act_dim = act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.scans = np.zeros((max_size, frames, laser_dim), dtype='float32')
        self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size,), dtype='float32')
        self.terminal = np.zeros((max_size,), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_scans = np.zeros((max_size, frames, laser_dim), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    # 抽样指定数量（batch_size）的经验
    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        obs = self.obs[batch_idx]
        scans = self.scans[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        next_scans = self.next_scans[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, scans, action, reward, next_obs, next_scans, terminal

    def append(self, obs, scans, act, reward, next_obs, next_scans, terminal):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.scans[self._curr_pos] = scans
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.next_scans[self._curr_pos] = next_scans
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size

class ReplayMemory3(object):
    def __init__(self, max_size, obs_dim, laser_dim, frames, act_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.laser_dim = laser_dim
        self.act_dim = act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.scans = np.zeros((max_size, frames, laser_dim), dtype='float32')
        self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size,), dtype='float32')
        self.terminal = np.zeros((max_size,), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_scans = np.zeros((max_size, frames, laser_dim), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    # 抽样指定数量（batch_size）的经验
    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        obs = self.obs[batch_idx]
        scans = self.scans[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        next_scans = self.next_scans[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, scans, action, reward, next_obs, next_scans, terminal

    def append(self, obs, scans, act, reward, next_obs, next_scans, terminal):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.scans[self._curr_pos] = scans
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.next_scans[self._curr_pos] = next_scans
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size