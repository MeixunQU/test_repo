from setuptools import setup
import os
from glob import glob

package_name = 'multi_agent_sim'
# path = os.path.abspath(".")

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob("launch/*.launch.py")),
        # ('share/' + package_name, ['launch/start_train.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='meixun',
    maintainer_email='meixun@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = multi_agent_sim.train:main',
            'hyparams_server = multi_agent_sim.hyparams_server:main',
            'hyparams_client = multi_agent_sim.hyparams_client:main',
            'train_ros2 = multi_agent_sim.train_ros2:main',
            'rl_params_server = multi_agent_sim.rl_params_server:main'
        ],
    },
)
