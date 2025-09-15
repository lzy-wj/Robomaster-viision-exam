from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 声明启动参数
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('rm_vision_sim'),
            'config',
            'detector_config.yaml'
        ]),
        description='检测节点配置文件路径'
    )
    
    enable_debug_arg = DeclareLaunchArgument(
        'enable_debug',
        default_value='false',
        description='是否启用调试图像发布'
    )
    
    fps_monitor_arg = DeclareLaunchArgument(
        'fps_monitor',
        default_value='true', 
        description='是否启用帧率监控'
    )

    # 节点定义
    camera_node = Node(
        package='rm_vision_sim',
        executable='camera_node',
        name='camera_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'frame_rate': 10.0,
            'image_width': 640,
            'image_height': 480,
        }]
    )

    detector_node = Node(
        package='rm_vision_sim',
        executable='detector_node', 
        name='detector_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'enable_debug_image': LaunchConfiguration('enable_debug'),
                'enable_fps_monitor': LaunchConfiguration('fps_monitor'),
            }
        ]
    )

    logger_node = Node(
        package='rm_vision_sim',
        executable='logger_node',
        name='logger_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'log_level': 'info',
            'position_format': 'Target detected at: [x={:.1f}, y={:.1f}]'
        }]
    )

    viewer_node = Node(
        package='rm_vision_sim',
        executable='viewer_node',
        name='viewer_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'window_name': 'RM Viewer',
            'crosshair_color': [0, 255, 0],
            'show_fps': True,
        }]
    )

    return LaunchDescription([
        config_file_arg,
        enable_debug_arg,
        fps_monitor_arg,
        camera_node,
        detector_node,
        logger_node,
        viewer_node,
    ])


