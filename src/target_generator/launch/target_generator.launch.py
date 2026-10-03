from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='target_generator',
            executable='random_target',
            name='random_target',
            output='screen',
        ),

        Node(
            package='target_generator',
            executable='marker',
            name='marker',
            output='screen',
        ),
    ])
