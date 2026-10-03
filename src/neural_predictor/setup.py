from setuptools import find_packages, setup
import os 
package_name = 'neural_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    (
        'share/ament_index/resource_index/packages',
        ['resource/neural_predictor']
    ),
    (
        'share/neural_predictor',
        ['package.xml']
    ),
    (
        os.path.join('share', 'neural_predictor', 'launch'),
        [
            'launch/nn_robot.launch.py',
            'launch/main_ik_comparison.launch.py'
        ]
    ),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='glados',
    maintainer_email='silas.helgesson@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'online_training=neural_predictor.online_training_ik:main',
        'data_collector=neural_predictor.data_collector:main',
        ],
    },
)