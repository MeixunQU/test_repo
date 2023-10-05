import numpy as np
import math
import copy
# import matplotlib.pyplot as plt
# import cv2

h_f = 6
h_b = 2
w_l = 4
w_r = 4
assert h_f + h_b == w_l + w_r
# For a single row or column, the number of pixel dots used for quantization
dot_num = (w_l + w_r) * 100 + 1
delta_s = (w_l + w_r) / (dot_num - 1)
# print('resolution:', delta_s)
max_distance_front = math.sqrt((w_r**2)+(h_b**2))
max_distance_back = math.sqrt((w_r**2)+(h_f**2))
# print('math distance (y<0):', max_distance_front)
# print('math distance (y>0):', max_distance_back)


terminal_coord_I = '({}, {})'.format(math.floor(w_r/delta_s), math.floor(h_f/delta_s))
terminal_coord_II = '({}, {})'.format(math.ceil(-w_l/delta_s), math.floor(h_f/delta_s))
terminal_coord_III = '({}, {})'.format(math.ceil(-w_l/delta_s), math.ceil(-h_b/delta_s))
terminal_coord_IV = '({}, {})'.format(math.floor(w_r/delta_s), math.ceil(-h_b/delta_s))
# print('terminal coord (I):', terminal_coord_I)
# print('terminal coord (II):', terminal_coord_II)
# print('terminal coord (III):', terminal_coord_III)
# print('terminal coord (IV):', terminal_coord_IV)
# print()


fov = 4.7  # in rad, 269.29 degree
num_beams = 1080

increment_angle = fov / (num_beams - 1)
increment_angle_degree = increment_angle * 180/np.pi
first_scan_angle = -fov / 2 + np.pi/2
first_scan_degree = first_scan_angle * 180/np.pi
num_179_scan_degree = first_scan_degree + (179-1) * increment_angle_degree
num_180_scan_degree = first_scan_degree + (180-1) * increment_angle_degree
num_540_scan_degree = first_scan_degree + (540-1) * increment_angle_degree
num_541_scan_degree = first_scan_degree + (541-1) * increment_angle_degree
num_901_scan_degree = first_scan_degree + (901-1) * increment_angle_degree
num_902_scan_degree = first_scan_degree + (902-1) * increment_angle_degree
last_scan_angle = first_scan_angle + (num_beams - 1) * increment_angle
last_scan_degree = last_scan_angle * 180/np.pi

# print('increment angle degree:', increment_angle_degree)
# print('1# scan degree:', first_scan_degree)
# print('179# scan degree:', num_179_scan_degree)
# print('180# scan degree:', num_180_scan_degree)
# print('540# scan degree:', num_540_scan_degree)
# print('541# scan degree:', num_541_scan_degree)
# print('901# scan degree:', num_901_scan_degree)
# print('902# scan degree:', num_902_scan_degree)
# print('1080# scan degree:', last_scan_degree)

#  group division
# 1# - 179#, 902# - 1080# (y<0)
# 180# - 901# (y>0)


initial_laser_list = []
for i in range(num_beams):
    # start degree: (-4.7/6)
    angle = first_scan_angle + i * increment_angle
    initial_laser_list.append([angle, -1])

# print(initial_laser_list)


# RADIUS = 1.5
# def lidar_analysis(obs):
#     v = obs['linear_vels_x'][0]
#     heading_angle = obs['poses_theta'][0]
#
#     laser_list = obs['scans'][0]
#     polar_laser_list = copy.deepcopy(initial_laser_list)
#     for i in range(len(laser_list)):
#         polar_laser_list[i][1] = laser_list[i]
#
#     refined_laser_distance_list = []
#     refined_ttc_list = []
#     for laser_info in polar_laser_list:
#         # if laser_info[1] < RADIUS and np.cos(heading_angle - laser_info[0]) > 0.1:
#         #     # print(v, np.cos(heading_angle - laser_info[0]))
#         #     ttc = laser_info[1] / (v * np.cos(heading_angle - laser_info[0]))
#         #     refined_ttc_list.append(ttc)
#         #     refined_laser_distance_list.append(laser_info[1])
#         if laser_info[1] < RADIUS:
#             # print(v, np.cos(heading_angle - laser_info[0]))
#             ttc = laser_info[1] / (v * np.cos(heading_angle - laser_info[0]))
#             refined_ttc_list.append(ttc)
#             refined_laser_distance_list.append(laser_info[1])
#
#     if len(refined_laser_distance_list) > 0:
#         obstacle_detected = 1
#         # avg_ttc = np.median(refined_ttc_list)
#         # min_ttc = min(refined_ttc_list)
#         avg_distance = np.mean(refined_laser_distance_list)
#     else:
#         obstacle_detected = 0
#         avg_ttc = -1
#         min_ttc = -1
#         avg_distance = 0
#
#     return obstacle_detected, avg_distance


def lidar_analysis(obs, RADIUS):
    polar_laser_list = copy.deepcopy(initial_laser_list)
    heading_angle = obs['poses_theta'][0]
    laser_list = obs['scans'][0]

    for i in range(len(laser_list)):
        polar_laser_list[i][1] = laser_list[i]

    refined_laser_distance_list = []
    for laser_info in polar_laser_list:
        # if laser_info[1] < RADIUS and np.cos(heading_angle - laser_info[0]) > 0:
        if laser_info[1] < RADIUS:
            refined_laser_distance_list.append(laser_info[1])

    if len(refined_laser_distance_list) > 0:
        obstacle_danger = True
        num_obstacle = len(refined_laser_distance_list)
        avg_distance = np.mean(refined_laser_distance_list)
        min_distance = min(laser_list)
    else:
        obstacle_danger = False
        num_obstacle = 0
        avg_distance = -1
        min_distance = min(laser_list)

    return obstacle_danger, num_obstacle, avg_distance, min_distance


def lidar_to_2D(laser_list, total_step):
    polar_laser_list = copy.deepcopy(initial_laser_list)
    for i in range(len(laser_list)):
        polar_laser_list[i][1] = laser_list[i]
    # print(polar_laser_list)

    cartesian_coords_list = []
    cartesian_laser_list = []
    for i in range(len(polar_laser_list)):
        theta = polar_laser_list[i][0]
        distance = polar_laser_list[i][1]
        quantize_x = math.floor(distance * np.cos(theta) / delta_s)
        quantize_y = math.floor(distance * np.sin(theta) / delta_s)

        # IV, x > 0 and y < 0
        if 0 <= i <= 178:
            if 0 <= quantize_x <= math.floor(w_r / delta_s) and math.ceil(-h_b / delta_s) <= quantize_y <= 0:
                if [quantize_x, quantize_y] not in cartesian_coords_list:
                    cartesian_coords_list.append([quantize_x, quantize_y])
                    cartesian_laser_list.append([quantize_x, quantize_y, distance, 1])
                else:
                    index = cartesian_coords_list.index([quantize_x, quantize_y])
                    cartesian_laser_list[index][2] = cartesian_laser_list[index][2] + 1
                    cartesian_laser_list[index][3] = cartesian_laser_list[index][3] + 1

        # I, x > 0 and y > 0
        if 179 <= i <= 539:
            if 0 <= quantize_x <= math.floor(w_r / delta_s) and 0 <= quantize_y <= math.floor(h_f / delta_s):
                if [quantize_x, quantize_y] not in cartesian_coords_list:
                    cartesian_coords_list.append([quantize_x, quantize_y])
                    cartesian_laser_list.append([quantize_x, quantize_y, distance, 1])
                else:
                    index = cartesian_coords_list.index([quantize_x, quantize_y])
                    cartesian_laser_list[index][2] = cartesian_laser_list[index][2] + 1
                    cartesian_laser_list[index][3] = cartesian_laser_list[index][3] + 1

        # II, x > 0 and y < 0
        if 540 <= i <= 900:
            if math.ceil(-w_l / delta_s) <= quantize_x <= 0 and 0 <= quantize_y <= math.floor(h_f / delta_s):
                if [quantize_x, quantize_y] not in cartesian_coords_list:
                    cartesian_coords_list.append([quantize_x, quantize_y])
                    cartesian_laser_list.append([quantize_x, quantize_y, distance, 1])
                else:
                    index = cartesian_coords_list.index([quantize_x, quantize_y])
                    cartesian_laser_list[index][2] = cartesian_laser_list[index][2] + 1
                    cartesian_laser_list[index][3] = cartesian_laser_list[index][3] + 1

        # III, x < 0 and y < 0
        else:
            if math.ceil(-w_l / delta_s) <= quantize_x <= 0 and math.ceil(-h_b / delta_s) <= quantize_y <= 0:
                if [quantize_x, quantize_y] not in cartesian_coords_list:
                    cartesian_coords_list.append([quantize_x, quantize_y])
                    cartesian_laser_list.append([quantize_x, quantize_y, distance, 1])
                else:
                    index = cartesian_coords_list.index([quantize_x, quantize_y])
                    cartesian_laser_list[index][2] = cartesian_laser_list[index][2] + 1
                    cartesian_laser_list[index][3] = cartesian_laser_list[index][3] + 1

    for i in range(len(cartesian_laser_list)):
        if cartesian_laser_list[i][3] > 1:
            # print(cartesian_laser_list[i])
            cartesian_laser_list[i][2] = cartesian_laser_list[i][2] / cartesian_laser_list[i][3]

    pixel_list = []
    for i in range(len(cartesian_laser_list)):
        original_x = cartesian_laser_list[i][0]
        original_y = cartesian_laser_list[i][1]

        new_x = original_x + math.floor(w_l / delta_s)
        new_y = -original_y + math.floor(h_f / delta_s)

        pixel_list.append([new_y, new_x, 0])

    return generate_img(pixel_list, total_step)


def generate_img(pixel_list, total_step):
    img = np.zeros((dot_num, dot_num), dtype=np.uint8)
    img = 255 - img

    for pixel_dot in pixel_list:
        y = pixel_dot[0]
        x = pixel_dot[1]
        pixel_value = pixel_dot[2]
        img[y][x] = pixel_value
    print('img.shape:', img.shape)
    img_shape = img.shape
    # cv2.imwrite('lidar_map/step_' + str(total_step) + '.png', img)

    img_flatten = img.flatten()
    print('img_flatten.shape:', img_flatten.shape)
    return img_flatten, img_shape
