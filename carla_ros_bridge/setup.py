"""
Setup for carla_ros_bridge
"""
import os
from glob import glob
ROS_VERSION = int(os.environ['ROS_VERSION'])

if ROS_VERSION == 1:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(packages=['carla_ros_bridge'], package_dir={'': 'src'})

    setup(**d)

elif ROS_VERSION == 2:
    from setuptools import setup

    package_name = 'carla_ros_bridge'
    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name],
        data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                    ('share/' + package_name, ['package.xml']),
                    (os.path.join('share', package_name), glob('launch/*.launch.py')),
                    (os.path.join('share', package_name + '/test'), glob('test/test_objects.json'))],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='CARLA Simulator Team',
        maintainer_email='carla.simulator@gmail.com',
        description='CARLA ROS2 bridge',
        license='MIT',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': ['bridge = carla_ros_bridge.bridge:main',
            'carla_path_planning_plus1 = carla_ros_bridge.carla_path_planning_plus1:main',
            'carla_path_planning_plus2 = carla_ros_bridge.carla_path_planning_plus2:main',
            'carla_path_planning_plus3 = carla_ros_bridge.carla_path_planning_plus3:main',
            'carla_path_planning_plus5 = carla_ros_bridge.carla_path_planning_plus5:main',
            'carla_path_planning = carla_ros_bridge.carla_path_planning:main',
            'carla_obstacle_avoidance = carla_ros_bridge.carla_obstacle_avoidance_v2:main',
            'carla_old_obstacle_avoidance = carla_ros_bridge.carla_obstacle_avoidance:main',
            'carla_throttle_node = carla_ros_bridge.carla_throttle_node:main',
            'carla_throttle_node_v2 = carla_ros_bridge.carla_throttle_node_v2:main',
            'carla_segnode = carla_ros_bridge.carla_segnode:main',
            'carla_odom_relay = carla_ros_bridge.carla_odom_relay:main',
            'carla_steering_throttle_control = carla_ros_bridge.carla_steering_throttle_control:main',
            'carla_stop_node = carla_ros_bridge.carla_stop_node:main',
            'waypoints_eloborate= carla_ros_bridge.waypoints_eloborate:main',
            'waypoints = carla_ros_bridge.waypoints:main'],
        },
        package_dir={'': 'src'},
        package_data={'': ['CARLA_VERSION']},
        include_package_data=True
    )
