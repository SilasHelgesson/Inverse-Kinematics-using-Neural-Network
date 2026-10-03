import os

from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    franka_bringup = get_package_share_directory('franka_bringup')

    robot_description = ParameterValue(
        Command([
            'xacro ',
            os.path.join(
                franka_bringup,
                'urdf',
                'franka_arm.urdf.xacro'
            ),
            ' robot_type:=fr3',
        ]),
        value_type=str
    )

    return LaunchDescription([

        # Second robot_state_publisher for the neural network robot
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='nn_robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'robot_description': robot_description,
                    'frame_prefix': 'nn/',
                }
            ],
            remappings=[
                ('joint_states', '/network_joints'),
            ],
        ),

        # Connect the NN robot's root frame to the normal robot's base
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='nn_base_tf',
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--roll', '0',
                '--pitch', '0',
                '--yaw', '0',
                '--frame-id', 'base',
                '--child-frame-id', 'nn/base',
            ],
            output='screen',
        ),
    ])