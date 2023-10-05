# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Rendering engine for f1tenth gym env based on pyglet and OpenGL
Author: Hongrui Zheng
"""

# opengl stuff
import pyglet
from pyglet.gl import *
# from pyglet import shapes

# other
import numpy as np
from PIL import Image
import yaml

# helpers
# from envs.collision_models import get_vertices
from collision_models import get_vertices

# zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

# target shape
TARGET_LENGTH = 0.4
TARGET_WIDTH = 0.4


class EnvRenderer(pyglet.window.Window):
    """
    A window class inherited from pyglet.window.Window, handles the camera/projection interaction, resizing window, and rendering the environment
    """

    def __init__(self, width, height, *args, **kwargs):
        # def __init__(self, width, height, target_x, target_y, *args, **kwargs):
        """
        Class constructor

        Args:
            width (int): width of the window
            height (int): height of the window

        Returns:
            None
        """
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf, resizable=True, vsync=False, *args, **kwargs)
        # super().__init__(width, height, target_x, target_y, config=conf, resizable=True, vsync=False, *args, **kwargs)

        # gl init
        glClearColor(9 / 255, 32 / 255, 87 / 255, 1.)

        # initialize camera values
        self.left = -width / 2
        self.right = width / 2
        self.bottom = -height / 2
        self.top = height / 2
        self.zoom_level = 1.2
        self.zoomed_width = width
        self.zoomed_height = height

        # current batch that keeps track of all graphics
        self.batch = pyglet.graphics.Batch()

        # current env map
        self.map_points = None

        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        self.poses = None
        # self.target_np = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        self.vertices = None
        # self.target_vertices = None

        # current score label
        # self.score_label = pyglet.text.Label(
        #         'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(
        #             laptime=0.0, count=0.0),
        #         font_size=36,
        #         x=0,
        #         y=-800,
        #         anchor_x='center',
        #         anchor_y='center',
        #         # width=0.01,
        #         # height=0.01,
        #         color=(255, 255, 255, 255),
        #         batch=self.batch)

        self.target_render = pyglet.text.Label(
            'target x: {target_x}, target y: {target_y}'.format(
                target_x=0.0, target_y=0.0),
            font_size=36,
            x=700,
            y=-200,
            anchor_x='center',
            anchor_y='center',
            # width=0.01,
            # height=0.01,
            color=(255, 255, 255, 255),
            batch=self.batch)

        self.fps_display = pyglet.window.FPSDisplay(self)

        # self.target_position_np = get_vertices(np.array([0.0, 0.0, 0.0]), TARGET_LENGTH, TARGET_LENGTH)
        #
        # self.target_vertices = list(self.target_position_np.flatten())
        # # print(target_vertices)
        # self.target = self.batch.add(4, GL_QUADS, None, ('v2f', self.target_vertices),
        #                     ('c3B', [135, 206, 250, 135, 206, 250, 135, 206, 250, 135, 206, 250]))

    def update_map(self, map_path, map_ext):
        """
        Update the map being drawn by the renderer. Converts image to a list of 3D points representing each obstacle pixel in the map.

        Args:
            map_path (str): absolute path to the map without extensions
            map_ext (str): extension for the map image file

        Returns:
            None
        """

        # load map metadata
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']
                origin = map_metadata['origin']
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        map_img = np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x * map_resolution + origin_x).flatten()
        print('x coordinate of map:', map_x)
        map_y = (map_y * map_resolution + origin_y).flatten()
        print('y coordinate of map:', map_y)
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))

        # mask and only leave the obstacle points
        map_mask = map_img == 0.0
        map_mask_flat = map_mask.flatten()
        map_points = 50. * map_coords[:, map_mask_flat].T
        # print(map_points)
        # render the map image (.png)
        for i in range(map_points.shape[0]):
            self.batch.add(1, GL_POINTS, None, ('v3f/stream', [map_points[i, 0], map_points[i, 1], map_points[i, 2]]),
                           ('c3B/stream', [183, 193, 222]))
        self.map_points = map_points

    def on_resize(self, width, height):
        """
        Callback function on window resize, overrides inherited method, and updates camera values on top of the inherited on_resize() method.

        Potential improvements on current behavior: zoom/pan resets on window resize.

        Args:
            width (int): new width of window
            height (int): new height of window

        Returns:
            None
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.left = -self.zoom_level * width / 2
        self.right = self.zoom_level * width / 2
        self.bottom = -self.zoom_level * height / 2
        self.top = self.zoom_level * height / 2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Callback function on mouse drag, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            dx (int): Relative X position from the previous mouse position.
            dy (int): Relative Y position from the previous mouse position.
            buttons (int): Bitwise combination of the mouse buttons currently pressed.
            modifiers (int): Bitwise combination of any keyboard modifiers currently active.

        Returns:
            None
        """

        # pan camera
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Callback function on mouse scroll, overrides inherited method.

        Args:
            x (int): Distance in pixels from the left edge of the window.
            y (int): Distance in pixels from the bottom edge of the window.
            scroll_x (float): Amount of movement on the horizontal axis.
            scroll_y (float): Amount of movement on the vertical axis.

        Returns:
            None
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if .01 < self.zoom_level * f < 10:
            self.zoom_level *= f

            (width, height) = self.get_size()

            mouse_x = x / width
            mouse_y = y / height

            mouse_x_in_world = self.left + mouse_x * self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y * self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self):
        """
        Callback function when the 'x' is clicked on the window, overrides inherited method. Also throws exception to end the python program when in a loop.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: with a message that indicates the rendering window was closed
        """

        super().on_close()
        raise Exception('Rendering window was closed.')

    def on_draw(self):
        """
        Function when the pyglet is drawing. The function draws the batch created that includes the map points, the agent polygons, and the information text, and the fps display.
        
        Args:
            None

        Returns:
            None
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception('Map not set for renderer.')
        if self.poses is None:
            raise Exception('Agent poses not updated for renderer.')

        # Initialize Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set orthographic projection matrix
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        # Draw all batches
        # print('here is to run batch draw ')
        self.batch.draw()
        self.fps_display.draw()
        # Remove default modelview matrix
        glPopMatrix()

    def update_obs(self, obs):
        """
        Updates the renderer with the latest observation from the gym environment, including the agent poses, and the information text.

        Args:
            obs (dict): observation dict from the gym env

        Returns:
            None
        """
        # print('obs in update_obs for rendering', obs)
        # obs -> {'ego_idx': 0, 'poses_x': [0.0], 'poses_y': [0.0], 'poses_theta': [1.57], 'lap_times': array([0.02]),
        # 'lap_counts': array([0.]), 'target_position_x': -10. 0, 'target_position_y': -3.0}

        # print('-------------------------------------------')
        self.ego_idx = obs['ego_idx']
        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']
        # targets_x = obs['target_position_x']
        # targets_y = obs['target_position_y']
        targets = obs['target_position']
        num_agents = len(poses_x)

        if self.poses is None:
            self.cars = []
            self.targets = []
            for i in range(num_agents):
                if i == 0:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    # print('vertices:', vertices)
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices),
                                         ('c3B', [172, 97, 185, 172, 97, 185, 172, 97, 185, 172, 97, 185]))
                    self.cars.append(car)

                    target_position_np = 50. * get_vertices(np.array([0.0, 0.0, 0.0]), TARGET_LENGTH, TARGET_LENGTH)
                    # print('target_position_np', target_position_np)
                    target_vertices = list(target_position_np.flatten())
                    # print(target_vertices)
                    target = self.batch.add(4, GL_QUADS, None, ('v2f', target_vertices),
                                            ('c3B', [135, 206, 250, 135, 206, 250, 135, 206, 250, 135, 206, 250]))
                    self.targets.append(target)
                elif i == 1:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    c = 100 * 0
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices),
                                         ('c3B', [99+c, 52+c, 94+c, 99+c, 52+c, 94+c, 99+c, 52+c, 94+c, 99+c, 52+c, 94+c]))
                    self.cars.append(car)

                    target_position_np = 50. * get_vertices(np.array([0.0, 0.0, 0.0]), TARGET_LENGTH, TARGET_LENGTH)
                    # print('target_position_np', target_position_np)
                    target_vertices = list(target_position_np.flatten())
                    # print(target_vertices)
                    d = 100 * 1
                    target = self.batch.add(4, GL_QUADS, None, ('v2f', target_vertices),
                                            ('c3B', [135+d, 206+d, 250+d, 135+d, 206+d, 250+d, 135+d, 206+d, 250+d, 135+d, 206+d, 250+d]))
                    self.targets.append(target)
                elif i == 2:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    c = 100 * 1
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices),
                                         ('c3B', [99+c, 52+c, 94+c, 99+c, 52+c, 94+c, 99+c, 52+c, 94+c, 99+c, 52+c, 94+c]))
                    self.cars.append(car)

                    target_position_np = 50. * get_vertices(np.array([0.0, 0.0, 0.0]), TARGET_LENGTH, TARGET_LENGTH)
                    # print('target_position_np', target_position_np)
                    target_vertices = list(target_position_np.flatten())
                    # print(target_vertices)
                    d = 100 * 2
                    target = self.batch.add(4, GL_QUADS, None, ('v2f', target_vertices),
                                            ('c3B', [135+d, 206+d, 250+d, 135+d, 206+d, 250+d, 135+d, 206+d, 250+d, 135+d, 206+d, 250+d]))
                    self.targets.append(target)
                else:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    c = 100 * 2
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices),
                                         ('c3B', [99+c, 52+c, 94+c, 99+c, 52+c, 94+c, 99+c, 52+c, 94+c, 99+c, 52+c, 94+c]))
                    self.cars.append(car)

                    target_position_np = 50. * get_vertices(np.array([0.0, 0.0, 0.0]), TARGET_LENGTH, TARGET_LENGTH)
                    # print('target_position_np', target_position_np)
                    target_vertices = list(target_position_np.flatten())
                    # print(target_vertices)
                    d = 100 * 3
                    target = self.batch.add(4, GL_QUADS, None, ('v2f', target_vertices),
                                            ('c3B', [135+d, 206+d, 250+d, 135+d, 206+d, 250+d, 135+d, 206+d, 250+d, 135+d, 206+d, 250+d]))
                    self.targets.append(target)


        # print('self.cars:', self.cars)
        # print('self.targets:', self.targets)

        poses = np.stack((poses_x, poses_y, poses_theta)).T
        # targets = np.stack([(targets_x, targets_y, 0)])
        # -> dimension of the array poses
        # print('poses for car:', poses.shape)

        # print('obs:', obs)
        # print('poses:', poses)    #  -> [[ 9.34586730e-07 -4.82240927e-03  1.57299922e+00]]
        # print('poses[0, :]:', poses[0, :])
        # print('targets:', targets)
        # print('targets[0, :]:', targets[0, :])

        for j in range(poses.shape[0]):
            # print(poses.shape[0])
            print('poses[j, :]', poses[j, :])
            vertices_np = 50. * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            # print('cars vertices', vertices_np)
            vertices = list(vertices_np.flatten())
            # print('cars vertices:', vertices)
            self.cars[j].vertices = vertices
        self.poses = poses

        # print(self.cars)
        # print()
        # print(self.targets)
        # print('----------------------')
        for j in range(targets.shape[0]):
            # print(j)
            # print(targets.shape[0])
            # print('targets[j, :]', targets[j, :])
            vertices_np = 50. * get_vertices(targets[j, :], TARGET_LENGTH, TARGET_WIDTH)
            # print('targets vertices', vertices_np)
            vertices = list(vertices_np.flatten())
            # print('targets vertices:', vertices)
            self.targets[j].vertices = vertices
        # self.targets = targets

        # self.target_render.text = 'target_1 x: {target_1_x}, target_1 y: {target_1_y}, ' \
        #                           'target_2 x: {target_2_x}, target_2 y: {target_2_y}' \
        #                           ''.format(target_1_x=round(obs['target_position'][0][0], 2),
        #                                     target_1_y=round(obs['target_position'][0][1], 2),
        #                                     target_2_x=round(obs['target_position'][1][0], 2),
        #                                     target_2_y=round(obs['target_position'][1][1], 2))
        # print('-------------------------------------------')
