import torch
import numpy as np
import matplotlib.pyplot as plt

model_list = ['sac']
seed_list = [21]

# episode reward
sac_episode_reward_list = []
sac_evaluate_reward_list = []


def find_mean_min_max(input_list):
    compare_times = len(input_list[0])
    mean_list = []
    min_list = []
    max_list = []
    temp_list = []

    for i in range(compare_times):
        for j in range(len(input_list)):
            temp_list.append(input_list[j][i])
        mean_list.append(np.mean(temp_list))
        min_list.append(min(temp_list))
        max_list.append(max(temp_list))
        temp_list = []

    return [mean_list, min_list, max_list]


def plot_reward(sac_reward_mean_min_max_list):
    s = np.arange(len(sac_reward_mean_min_max_list[0]))
    plt.plot(s, sac_reward_mean_min_max_list[0], 'b-', label='SAC')
    plt.fill_between(s, sac_reward_mean_min_max_list[1], sac_reward_mean_min_max_list[2], color='#aabbcc', alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Episode reward')

    plt.legend()
    plt.grid()
    plt.savefig('results/sac/train/' + str(seed) + '/training_reward.pdf', bbox_inches='tight')
    plt.show()


def plot_evaluate(sac_evaluate_mean_min_max_list):
    s = np.arange(len(sac_evaluate_mean_min_max_list[0])) * 10
    plt.plot(s, sac_evaluate_mean_min_max_list[0], 'b-', label='SAC')
    plt.fill_between(s, sac_evaluate_mean_min_max_list[1], sac_evaluate_mean_min_max_list[2], color='#aabbcc', alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Evaluate reward')

    plt.legend()
    plt.grid()
    plt.savefig('results/sac/train/' + str(seed) + '/evaluate_reward.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # load data
    for model in model_list:
        for seed in seed_list:
            if model == 'sac':
                path = 'results/sac/train/' + str(seed) + '/info_dict'
                info_dict = torch.load(path)
                sac_episode_reward_list.append(info_dict['episode_reward_list'])
                sac_evaluate_reward_list.append(info_dict['evaluate_reward_list'])
                print(info_dict['episode_reward_list'])
                print('best_episode_reward:', info_dict['best_episode_reward'])
                print('best_episode_index:', info_dict['best_episode_index'])

print('plot training episode reward')
sac_reward_mean_min_max_list = find_mean_min_max(sac_episode_reward_list)
plot_reward(sac_reward_mean_min_max_list)

print('plot evaluate episode reward')
sac_evaluate_mean_min_max_list = find_mean_min_max(sac_evaluate_reward_list)
plot_evaluate(sac_evaluate_mean_min_max_list)

# episode_num = len(info_dict['episode_reward_list'])
# episode = []
# for i in range(episode_num):
#     episode.append(i)
# plt.plot(episode, info_dict['episode_reward_list'])
# plt.show()
