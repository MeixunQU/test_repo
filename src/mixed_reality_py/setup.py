from setuptools import setup

package_name = 'mixed_reality_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'sim2pub = mixed_reality_py.sim2pub:main',
            'sub_pose_realcar = mixed_reality_py.sub_pose_realcar:main'
        ],
    },
)
