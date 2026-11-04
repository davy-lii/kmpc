#!/usr/bin/env python3
"""Smoothly move the right arm to a target through joint stream control."""

import argparse
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from xoip_ros2.msg import JointStreamCommand


COMMAND_TOPIC = '/xoip/robot/joint_stream_cmd'
JOINT_STATE_TOPIC = '/joint_states'
ROBOT_ID = 1
# SDK 汇总名；Thor 聚合 /joint_states 常用 URDF 名。两种都认。
RIGHT_JOINT_NAMES_SDK = [f'joint_{index}' for index in range(22, 29)]
RIGHT_JOINT_NAMES_URDF = [f'RightArm_Z_Joint{i}' for i in range(1, 8)]
MAX_STEP_DEG = 0.1
FINAL_HOLD_SEC = 0.2

# 修改这里即可设置右臂目标位置，单位：度，顺序为 J1 到 J7。
RIGHT_TARGET_JOINTS_DEG = [-30.0, -39.38, 5.0, -52.30, 5.0, 0.0, 0.0]

# RIGHT_TARGET_JOINTS_DEG = [30.0, -34.38, 0.0, -57.30, 0.0, 0.0, 5.0]


# [-30.0, -34.38, 0.0, -57.3, 0.0, 0.0, 0.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='读取 /joint_states 的右臂当前位置并平滑插值到目标位置。'
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='确认向真机发布；不加此参数只计算和显示轨迹。',
    )
    parser.add_argument('--rate', type=float, default=250.0, help='发布频率，默认 250 Hz。')
    parser.add_argument(
        '--state-timeout', type=float, default=5.0,
        help='等待 /joint_states 的超时时间，默认 5 秒。',
    )
    return parser.parse_args()


def validate_args(args):
    if not math.isfinite(args.rate) or args.rate < 20.0:
        raise ValueError('rate 必须是至少 20 Hz 的有限数，以避免流控制超时。')
    if not math.isfinite(args.state_timeout) or args.state_timeout <= 0.0:
        raise ValueError('state-timeout 必须是有限正数。')
    if len(RIGHT_TARGET_JOINTS_DEG) != 7:
        raise ValueError('RIGHT_TARGET_JOINTS_DEG 必须包含 7 个关节角。')
    if not all(math.isfinite(value) for value in RIGHT_TARGET_JOINTS_DEG):
        raise ValueError('目标关节角必须全部是有限数。')


def extract_right_arm_degrees(msg):
    positions_by_name = dict(zip(msg.name, msg.position))
    used = None
    for names in (RIGHT_JOINT_NAMES_URDF, RIGHT_JOINT_NAMES_SDK):
        if all(name in positions_by_name for name in names):
            used = names
            break
    if used is None:
        raise ValueError(
            '/joint_states 缺少右臂关节。需要 RightArm_Z_Joint1..7 或 joint_22..28，'
            f'实际 name={list(msg.name)[:20]}...'
        )

    # 汇总 /joint_states 按 ROS 约定使用弧度；joint_stream_cmd 使用度。
    radians = [float(positions_by_name[name]) for name in used]
    if not all(math.isfinite(value) for value in radians):
        raise ValueError('/joint_states 中包含非有限关节值。')
    print(f'右臂关节名: {used[0]} .. {used[-1]}')
    return [math.degrees(value) for value in radians]


def build_trajectory(current_deg, target_deg):
    deltas = [target - current for current, target in zip(current_deg, target_deg)]
    max_delta = max(abs(delta) for delta in deltas)
    segment_count = max(1, math.ceil(max_delta / MAX_STEP_DEG))
    trajectory = [
        [current + delta * step / segment_count for current, delta in zip(current_deg, deltas)]
        for step in range(1, segment_count + 1)
    ]
    return trajectory, max_delta


def wait_for_joint_state(node, timeout_sec, received_after=None):
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_joint_state is not None:
            if received_after is None or node.latest_joint_state_time >= received_after:
                return node.latest_joint_state
    raise TimeoutError(f'{timeout_sec:g} 秒内没有收到 {JOINT_STATE_TOPIC}')


def spin_until(node, target_time):
    while rclpy.ok():
        remaining = target_time - time.monotonic()
        if remaining <= 0.0:
            return
        rclpy.spin_once(node, timeout_sec=min(remaining, 0.005))


def print_plan(current_deg, trajectory, rate):
    target_deg = trajectory[-1]
    max_step = max(
        abs(after - before)
        for before, after in zip(current_deg, trajectory[0])
    )
    if len(trajectory) > 1:
        max_step = max(
            max(abs(after - before) for before, after in zip(previous, current))
            for previous, current in zip(trajectory, trajectory[1:])
        )
    print(f'当前位置 (deg): {[round(value, 4) for value in current_deg]}')
    print(f'目标位置 (deg): {[round(value, 4) for value in target_deg]}')
    print(f'插值点数: {len(trajectory)}，最大相邻角差: {max_step:.6f}°')
    print(f'发布频率: {rate:g} Hz，预计运动下发时间: {len(trajectory) / rate:.3f} 秒')


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f'参数错误：{exc}', file=sys.stderr)
        return 2

    os.environ.setdefault('ROS_DOMAIN_ID', '20')
    rclpy.init()
    node = Node('right_arm_smooth_reset_stream_publisher')
    node.latest_joint_state = None
    node.latest_joint_state_time = 0.0

    def joint_state_callback(msg):
        node.latest_joint_state = msg
        node.latest_joint_state_time = time.monotonic()

    node.joint_state_subscription = node.create_subscription(
        JointState, JOINT_STATE_TOPIC, joint_state_callback, 1
    )
    publisher = node.create_publisher(JointStreamCommand, COMMAND_TOPIC, 10)

    try:
        initial_state = wait_for_joint_state(node, args.state_timeout)
        current_deg = extract_right_arm_degrees(initial_state)
        trajectory, _ = build_trajectory(current_deg, RIGHT_TARGET_JOINTS_DEG)

        print(f'状态话题: {JOINT_STATE_TOPIC}')
        print(f'指令话题: {COMMAND_TOPIC}')
        print_plan(current_deg, trajectory, args.rate)
        if not args.execute:
            print('预览模式：未发布。确认环境安全后，加 --execute 执行。')
            return 0

        discovery_deadline = time.monotonic() + 3.0
        while publisher.get_subscription_count() == 0 and time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if publisher.get_subscription_count() == 0:
            node.get_logger().error(f'{COMMAND_TOPIC} 没有订阅者，已取消复位。')
            return 1

        other_publishers = sorted({
            info.node_name
            for info in node.get_publishers_info_by_topic(COMMAND_TOPIC)
            if info.node_name != node.get_name()
        })
        if other_publishers:
            node.get_logger().warn(
                '检测到其他流式指令发布节点：' + ', '.join(other_publishers)
                + '；复位期间请停止其他控制。'
            )

        for seconds_left in (3, 2, 1):
            print(f'{seconds_left} 秒后开始复位；按 Ctrl-C 可取消。', flush=True)
            time.sleep(1.0)

        # 倒计时后重新获取一帧新状态，防止机器人在等待期间发生移动。
        fresh_after = time.monotonic()
        fresh_state = wait_for_joint_state(node, args.state_timeout, received_after=fresh_after)
        current_deg = extract_right_arm_degrees(fresh_state)
        trajectory, max_delta = build_trajectory(current_deg, RIGHT_TARGET_JOINTS_DEG)
        print_plan(current_deg, trajectory, args.rate)

        period = 1.0 / args.rate
        next_publish = time.monotonic()
        publish_count = 0
        for joints in trajectory:
            command = JointStreamCommand()
            command.header.stamp = node.get_clock().now().to_msg()
            command.robot_id = ROBOT_ID
            command.joints = [float(value) for value in joints]
            publisher.publish(command)
            publish_count += 1

            next_publish += period
            spin_until(node, next_publish)

        # 短暂保持最终目标，覆盖 DDS/调度造成的末帧丢失。
        final_command = JointStreamCommand()
        final_command.robot_id = ROBOT_ID
        final_command.joints = [float(value) for value in RIGHT_TARGET_JOINTS_DEG]
        hold_count = max(1, math.ceil(FINAL_HOLD_SEC * args.rate))
        for _ in range(hold_count):
            final_command.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(final_command)
            publish_count += 1
            next_publish += period
            spin_until(node, next_publish)

        print(
            f'平滑复位指令发送完成：最大总角差 {max_delta:.4f}°，'
            f'轨迹点 {len(trajectory)}，共发布 {publish_count} 条。'
        )
        return 0
    except (ValueError, TimeoutError) as exc:
        node.get_logger().error(str(exc))
        return 1
    except KeyboardInterrupt:
        print('\n已取消，停止发布复位指令。')
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
