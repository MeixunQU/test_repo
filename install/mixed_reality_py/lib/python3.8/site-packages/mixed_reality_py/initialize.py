import random
import numpy as np
# blockdict = {
#     # x range, y range
#     1: [(0.5, 5), (13, 15)],
#     2: [(0.5, 3.5), (8.5, 10.5)],
#     3: [(0.5, 9), (0.5, 6)],
#     4: [(5.5, 10), (6, 13)],
#     5: [(13.5, 18), (8, 15)],
#     6: [(12, 18.2), (0.5, 5.5)],
#     7: [(22, 25), (0.5, 15)]
# }


blockdict = {
    # x range, y range
    1: [(0.5+1.5, 5), (13+0.5, 15-1)],
    2: [(0.5+2, 3.5), (8.5+0.9, 10.5-1)],
    3: [(0.5+2, 9-1.5), (0.5+1.5, 6-1.5)],
    4: [(5.5+2, 10-1.5), (6+1.5, 13-2)],
    5: [(13.5+1.5, 18-1.5), (8+1.5, 15-1.5)],
    6: [(12+2, 18.2-1.5), (0.5+1.75, 5.5-1.5)],
    7: [(22+2, 25-1), (0.5+5, 15-5)]
}

def getPosition(blk_index):
    # pick up one position (x, y) in the given block
    x = random.uniform(blockdict[blk_index][0][0], blockdict[blk_index][0][1])
    y = random.uniform(blockdict[blk_index][1][0], blockdict[blk_index][1][1])
    return [x, y]


def initialize_episode():
    # blk_index_list = [1, 2, 3, 4, 5, 6, 7]
    while True:
        start_blk_index = random.choice([1, 2, 3, 4, 5, 6, 7])
        goal_blk_index = random.choice([1, 2, 3, 4, 5, 6, 7])
        # goal_blk_index = 1
        if start_blk_index != goal_blk_index:
            break

    start_pos = getPosition(start_blk_index)
    start_theta = random.uniform(0, 6.28)
    goal_pos = getPosition(goal_blk_index)

    return [start_pos[0], start_pos[1], start_theta], goal_pos


# pos_DEU116_list = [[-0.5303738117218018, 7.157588481903076],
#                    [2.222084856033325, 1.0575709342956543],
#                    [3.9719059467315674, 6.729456901550293]]

blockdict_DEU116 = {
    # x range, y range
    1: [(0.65, 2.8), (-0.15, 2.6)],
    2: [(-0.8, 1), (4.25, 8.2)],
    3: [(3.2, 4.1), (5.25, 8.25)],
    4: [(4.2, 5.4), (3.3, 4.4)]
}

def getPosition_DEU116(blk_index):
    # pick up one position (x, y) in the given block
    x = random.uniform(blockdict_DEU116[blk_index][0][0], blockdict_DEU116[blk_index][0][1])
    y = random.uniform(blockdict_DEU116[blk_index][1][0], blockdict_DEU116[blk_index][1][1])
    return [x, y]

def initialization():
    while True:
        start_blk_index = random.choice([1, 2, 3, 4])
        goal_blk_index = random.choice([1, 2, 3, 4])
        # goal_blk_index = 1
        if start_blk_index != goal_blk_index:
            break
    start_pos = getPosition_DEU116(start_blk_index)
    start_theta = random.uniform(0, 6.28)
    goal_pos = getPosition_DEU116(goal_blk_index)

    return [start_pos[0], start_pos[1], start_theta], goal_pos

blockdict_DEU116 = {
    # x range, y range
    1: [(0.65, 2.8), (-0.15, 2.6)],
    2: [(-0.8, 1), (4.25, 8.2)],
    3: [(3.2, 4.1), (5.25, 8.25)],
    4: [(4.2, 5.4), (3.3, 4.4)]
}

def getPosition_DEU116(blk_index):
    # pick up one position (x, y) in the given block
    x = random.uniform(blockdict_DEU116[blk_index][0][0], blockdict_DEU116[blk_index][0][1])
    y = random.uniform(blockdict_DEU116[blk_index][1][0], blockdict_DEU116[blk_index][1][1])
    return [x, y]

def initialize_episode2():

    start_pos_1 = [1.8, 0.8]
    start_theta_1 = random.uniform(0, 6.28)
    goal_pos_1 = [0.05, 6.8]
    # start_pos_2 = [0.05, 6.8]
    start_pos_2 = [2.108782406275817, 8.54741375474261]
    # start_theta_2 = random.uniform(0, 6.28)
    start_theta_2 = 0.5133899604152532
    goal_pos_2 = [1.8, 0.8]

    initial_list_1 = [[start_pos_1[0], start_pos_1[1], start_theta_1, goal_pos_1[0], goal_pos_1[1]],
                      [start_pos_2[0], start_pos_2[1], start_theta_2, goal_pos_2[0], goal_pos_2[1]]
                     ]
    initial_list = np.array(initial_list_1)
    return initial_list