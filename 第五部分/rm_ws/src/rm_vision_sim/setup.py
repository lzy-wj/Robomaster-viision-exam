from setuptools import setup

package_name = 'rm_vision_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/tracking_sim.launch.py']),
        ('share/' + package_name + '/config', ['config/detector_config.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rm',
    maintainer_email='rm@example.com',
    description='RM vision simulation with camera, detector, logger nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = rm_vision_sim.camera_node:main',
            'detector_node = rm_vision_sim.detector_node:main',
            'logger_node = rm_vision_sim.logger_node:main',
            'viewer_node = rm_vision_sim.viewer_node:main',
        ],
    },
)


