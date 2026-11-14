"""Shared launch helpers. Installed next to sim/real launch files."""

import os
import re

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _bool_cfg(context, name: str, default: str = 'true') -> bool:
    raw = context.launch_configurations.get(name, default)
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def python_share_paths():
    pkg = get_package_share_directory('whole_body_kmpc')
    return {
        'pkg': pkg,
        'xacro': os.path.join(pkg, 'urdf', 'Light-duty wheeled robot.urdf.xacro'),
        'params': os.path.join(pkg, 'config', 'kmpc_params.yaml'),
        'joint_map': os.path.join(pkg, 'config', 'joint_map.yaml'),
        'limits': os.path.join(pkg, 'config', 'joint_limits.yaml'),
        'q_pref': os.path.join(pkg, 'config', 'q_pref.yaml'),
        'sim_seed': os.path.join(pkg, 'config', 'sim_initial_joints.yaml'),
        'rviz': os.path.join(pkg, 'rviz', 'kmpc_sim.rviz'),
    }


def kmpc_node_action(robot_desc, paths, extra_params=None):
    params = {
        'robot_description': robot_desc,
        'joint_map_file': paths['joint_map'],
        'joint_limits_file': paths['limits'],
        'q_pref_file': paths['q_pref'],
        'kmpc_params_file': paths['params'],
        'auto_enable': ParameterValue(
            LaunchConfiguration('auto_enable'), value_type=bool),
        'demo_wave_waist': False,
        'dq_scale': ParameterValue(
            LaunchConfiguration('dq_scale'), value_type=float),
    }
    if extra_params:
        params.update(extra_params)
    return Node(
        package='kmpc_cpp',
        executable='kmpc_node',
        name='kmpc_node',
        output='screen',
        parameters=[paths['params'], params],
    )


def robot_state_publisher_action(robot_desc):
    return Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}],
    )


def _sanitize_robot_desc(robot_desc: str, xacro_path: str) -> str:
    """Make CAD URDF usable by robot_state_publisher / RViz.

    Relative meshes/foo.STL cannot be loaded from a parameter; point at
    files next to the xacro. Anonymous <material><color/></material> is
    dropped by urdf_parser (no name attribute).
    """
    mesh_dir = os.path.join(os.path.dirname(os.path.realpath(xacro_path)), 'meshes')
    if os.path.isdir(mesh_dir):
        robot_desc = robot_desc.replace(
            'filename="meshes/',
            f'filename="file://{mesh_dir}/',
        )
    robot_desc = re.sub(
        r'<material>\s*<color rgba="([^"]+)"\s*/>\s*</material>',
        r'<material name="cad"><color rgba="\1"/></material>',
        robot_desc,
    )
    return robot_desc


def rviz_action(rviz_config):
    return Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )


def _sim_setup(context, *args, **kwargs):
    paths = python_share_paths()
    sim_share = get_package_share_directory('robot_tf_sim')
    robot_desc = xacro.process_file(paths['xacro']).toxml()
    use_sim = _bool_cfg(context, 'use_sim')
    # 必须在 Include sim_tf 之前读：子 launch 也有 arg「rviz」，传入 false 会盖掉父配置。
    use_rviz = _bool_cfg(context, 'rviz')
    use_grasp = _bool_cfg(context, 'grasp', 'true')
    calibrate_first = _bool_cfg(context, 'calibrate_first', 'false')
    use_pose_gui = _bool_cfg(context, 'pose_gui', 'true')

    actions = []
    if use_sim:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(sim_share, 'launch', 'sim_tf.launch.py')),
                launch_arguments={
                    'rviz': 'false',
                    'publish_rate': '50.0',
                    'initial_positions_file': paths['sim_seed'],
                    'urdf_file': paths['xacro'],
                }.items(),
            )
        )

    actions.append(kmpc_node_action(robot_desc, paths, {
        'demo_wave_waist': ParameterValue(
            LaunchConfiguration('demo'), value_type=bool),
    }))
    if use_rviz:
        actions.append(rviz_action(paths['rviz']))

    if use_grasp:
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package='whole_body_kmpc',
                        executable='send_ee_target',
                        name='send_ee_target',
                        output='screen',
                        parameters=[{
                            'side': 'left',
                            'x': ParameterValue(
                                LaunchConfiguration('grasp_x'), value_type=float),
                            'y': ParameterValue(
                                LaunchConfiguration('grasp_y'), value_type=float),
                            'z': ParameterValue(
                                LaunchConfiguration('grasp_z'), value_type=float),
                            'frame_id': 'base_link',
                            'autostart': (not calibrate_first) and (not use_pose_gui),
                        }],
                    ),
                ],
            )
        )
    if use_pose_gui:
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package='whole_body_kmpc',
                        executable='pose_gui',
                        name='pose_gui',
                        output='screen',
                        parameters=[{
                            'q_pref_file': paths['q_pref'],
                            'q_pref_station': 'default',
                            'joint_limits_file': paths['limits'],
                        }],
                    ),
                ],
            )
        )
    if calibrate_first:
        actions.append(
            TimerAction(
                period=2.5,
                actions=[
                    Node(
                        package='whole_body_kmpc',
                        executable='calibrate_then_grasp',
                        name='calibrate_then_grasp',
                        output='screen',
                        parameters=[{
                            'q_pref_file': paths['q_pref'],
                            'q_pref_station': 'default',
                            'start_delay_sec': 0.5,
                            'autostart': True,
                        }],
                    ),
                ],
            )
        )
    return actions


def generate_sim_launch():
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('use_sim', default_value='true'),
        DeclareLaunchArgument('auto_enable', default_value='true'),
        DeclareLaunchArgument('dq_scale', default_value='1.0'),
        DeclareLaunchArgument('demo', default_value='false'),
        DeclareLaunchArgument('grasp', default_value='true'),
        DeclareLaunchArgument('calibrate_first', default_value='false'),
        DeclareLaunchArgument('pose_gui', default_value='true'),
        DeclareLaunchArgument('grasp_x', default_value='0.50'),
        DeclareLaunchArgument('grasp_y', default_value='0.12'),
        DeclareLaunchArgument('grasp_z', default_value='1.18'),
        OpaqueFunction(function=_sim_setup),
    ])


def _real_setup(context, *args, **kwargs):
    paths = python_share_paths()
    robot_desc = _sanitize_robot_desc(
        xacro.process_file(paths['xacro']).toxml(), paths['xacro'])
    use_rsp = _bool_cfg(context, 'use_rsp')
    use_rviz = _bool_cfg(context, 'rviz')
    actions = []
    if use_rsp:
        actions.append(robot_state_publisher_action(robot_desc))
    actions.append(kmpc_node_action(robot_desc, paths, {
        'line_dq_scale': ParameterValue(
            LaunchConfiguration('line_dq_scale'), value_type=float),
        'line_pace_mode': LaunchConfiguration('line_pace_mode'),
        'line_brake_distance_m': ParameterValue(
            LaunchConfiguration('line_brake_distance_m'), value_type=float),
        'line_brake_floor': ParameterValue(
            LaunchConfiguration('line_brake_floor'), value_type=float),
        'line_brake_profile': LaunchConfiguration('line_brake_profile'),
        'line_qp_max_iter': ParameterValue(
            LaunchConfiguration('line_qp_max_iter'), value_type=int),
        'stream_hz': ParameterValue(
            LaunchConfiguration('stream_hz'), value_type=float),
        'interp_catch_s': ParameterValue(
            LaunchConfiguration('interp_catch_s'), value_type=float),
        'interp_accel_s': ParameterValue(
            LaunchConfiguration('interp_accel_s'), value_type=float),
        'interp_jerk_s': ParameterValue(
            LaunchConfiguration('interp_jerk_s'), value_type=float),
        'stream_dedicated_thread': ParameterValue(
            LaunchConfiguration('stream_dedicated_thread'), value_type=bool),
        'stream_debug_q_hz': ParameterValue(
            LaunchConfiguration('stream_debug_q_hz'), value_type=float),
        'planner': LaunchConfiguration('planner'),
        'line_pose_pref_enable': ParameterValue(
            LaunchConfiguration('line_pose_pref_enable'), value_type=bool),
        'line_pitch_straighten_enable': ParameterValue(
            LaunchConfiguration('line_pitch_straighten_enable'), value_type=bool),
        'line_qp_pitch_monotonic_enable': ParameterValue(
            LaunchConfiguration('line_qp_pitch_monotonic_enable'), value_type=bool),
        'reverse_last_line_enable': ParameterValue(
            LaunchConfiguration('reverse_last_line_enable'), value_type=bool),
        'line_diag_enable': ParameterValue(
            LaunchConfiguration('line_diag_enable'), value_type=bool),
        'layer_diag_path': LaunchConfiguration('layer_diag_path'),
        'config_snapshot_path': LaunchConfiguration('config_snapshot_path'),
    }))
    if use_rviz:
        actions.append(rviz_action(paths['rviz']))
    return actions


def generate_real_launch():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='是否在本 launch 里起 rviz2（跟 kmpc_node 同一台机器）。Thor 容器默认关；笔记本上看姿态用 kmpc_cpp_viz'),
        DeclareLaunchArgument(
            'use_rsp', default_value='true',
            description='本机起 robot_state_publisher；驱动已发 TF 则关'),
        DeclareLaunchArgument('auto_enable', default_value='false'),
        DeclareLaunchArgument(
            'dq_scale', default_value='0.1',
            description='C10 真机第一档；只乘 Safety 之后的下发步'),
        DeclareLaunchArgument(
            'line_dq_scale', default_value=LaunchConfiguration('dq_scale'),
            description='直线段速度比例，默认跟随 dq_scale'),
        DeclareLaunchArgument(
            'line_pace_mode', default_value='r10b_first_order',
            description='直线路径 rate 更新：r10b_first_order（实验默认）或 r11_scurve（对照）'),
        DeclareLaunchArgument(
            'line_brake_distance_m', default_value='0.08',
            description='直线 path_remain 刹车距离（m）；与冻结 0.08 m 拆开，非法值拒绝启动'),
        DeclareLaunchArgument(
            'line_brake_floor', default_value='0.20',
            description='末端 rate 地板 (0,1]；旧硬编码 0.05 易慢尾'),
        DeclareLaunchArgument(
            'line_brake_profile', default_value='sqrt',
            description='末端降速曲线：sqrt（更快平均）或 linear（旧式）'),
        DeclareLaunchArgument(
            'line_qp_max_iter', default_value='200',
            description='直线 sparse ProxQP 最大迭代；非 SOLVED 仍拒绝该段'),
        DeclareLaunchArgument(
            'stream_hz', default_value='250.0',
            description='插补后发 multi_joint_servo_cmd 的频率；须与驱动 stream_control_rate 对齐'),
        DeclareLaunchArgument(
            'interp_catch_s', default_value='0.02',
            description='T_catch：走路用 max(catch, dt)；到点刹车看 s_stop'),
        DeclareLaunchArgument(
            'interp_accel_s', default_value='0.50',
            description='加到巡航速度的时间；a_bar = v_bar / accel_s；加大则到点刹车距更长'),
        DeclareLaunchArgument(
            'interp_jerk_s', default_value='0.08',
            description='0 到 a_bar 的 jerk 时间；S 曲线，消起步/刹车喀哒'),
        DeclareLaunchArgument(
            'stream_dedicated_thread', default_value='true',
            description='250 Hz 用独立线程 sleep_until 发流；false 回退 wall timer + 单线程 spin'),
        DeclareLaunchArgument(
            'stream_debug_q_hz', default_value='50.0',
            description='/kmpc/debug/q 频率；从 250 Hz 热路径降下来。0=不发'),
        DeclareLaunchArgument(
            'planner', default_value='unified_multistage_v2',
            description='单臂笛卡尔生产规划器；unified_multistage_v2 走 PlannerWorker+Follower+SafetySupervisor'),
        DeclareLaunchArgument(
            'line_pose_pref_enable', default_value='false',
            description='v2 下无决策权（保留参数兼容）；legacy 姿态偏好候选'),
        DeclareLaunchArgument(
            'line_pitch_straighten_enable', default_value='true',
            description='v2 下无决策权；legacy 名义路径俯仰单调化'),
        DeclareLaunchArgument(
            'line_qp_pitch_monotonic_enable', default_value='true',
            description='v2 下无决策权；legacy line_qp 俯仰单调约束'),
        DeclareLaunchArgument(
            'reverse_last_line_enable', default_value='true',
            description='v2 下无决策权；legacy reverse_cache 复用'),
        DeclareLaunchArgument(
            'line_diag_enable', default_value='false',
            description='启动晃动分层诊断环（默认关闭）'),
        DeclareLaunchArgument(
            'layer_diag_path', default_value='',
            description='分层诊断 JSONL 路径。空则不导出；须同时打开 line_diag_enable'),
        DeclareLaunchArgument(
            'config_snapshot_path', default_value='',
            description='有效配置快照 JSON 路径。空则只打到启动日志'),
        OpaqueFunction(function=_real_setup),
    ])


def _viz_setup(context, *args, **kwargs):
    paths = python_share_paths()
    robot_desc = _sanitize_robot_desc(
        xacro.process_file(paths['xacro']).toxml(), paths['xacro'])
    actions = [robot_state_publisher_action(robot_desc)]
    if _bool_cfg(context, 'rviz'):
        actions.append(rviz_action(paths['rviz']))
    return actions


def generate_viz_launch():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='只可视化：机器人 /joint_states → 本机 RViz，不起控制器'),
        OpaqueFunction(function=_viz_setup),
    ])
