#!/usr/bin/env python3
import os
import time
import unittest

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler,
    SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from std_msgs.msg import Float64, UInt32


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    width_arg = DeclareLaunchArgument('width', default_value='1920')
    height_arg = DeclareLaunchArgument('height', default_value='1080')
    use_cuda_arg = DeclareLaunchArgument('use_cuda', default_value='true')

    renderer_node = Node(
        package='tunnel_demo',
        executable='tunnel_renderer_node',
        name='tunnel_renderer',
        output='screen',
        parameters=[{
            'image_width': LaunchConfiguration('width'),
            'image_height': LaunchConfiguration('height'),
            'use_cuda': LaunchConfiguration('use_cuda'),
        }],
    )

    display_node = Node(
        package='tunnel_demo',
        executable='tunnel_display_node',
        name='tunnel_display',
        output='screen',
        parameters=[{'headless': True}],
    )

    rmw_zenohd = os.path.join(
        get_package_prefix('rmw_zenoh_cpp'), 'lib', 'rmw_zenoh_cpp', 'rmw_zenohd')
    zenoh_router = ExecuteProcess(
        cmd=[rmw_zenohd], name='zenoh_router', output='screen')

    ws_root = os.environ.get('PIXI_PROJECT_ROOT', '')
    if not ws_root:
        ws_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    libtorch_lib = os.path.join(ws_root, '.pixi', 'envs', 'default', 'libtorch', 'lib')
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if libtorch_lib not in ld_path:
        ld_path = f'{libtorch_lib}:{ld_path}' if ld_path else libtorch_lib

    return LaunchDescription([
        width_arg,
        height_arg,
        use_cuda_arg,
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
        SetEnvironmentVariable('LD_LIBRARY_PATH', ld_path),
        zenoh_router,
        RegisterEventHandler(
            OnProcessStart(
                target_action=zenoh_router,
                on_start=[
                    TimerAction(period=1.0, actions=[
                        renderer_node,
                        display_node,
                        launch_testing.actions.ReadyToTest(),
                    ]),
                ],
            ),
        ),
    ])


class TestTunnelDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_tunnel_demo')
        self.subscriber_count = 0
        self.latencies = []
        self.count_timestamps = []
        self.node.create_subscription(
            UInt32, 'subscriber_count', self._count_cb, 10)
        self.node.create_subscription(
            Float64, 'latency_ms', self._latency_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _count_cb(self, msg):
        self.subscriber_count = msg.data
        self.count_timestamps.append(time.time())

    def _latency_cb(self, msg):
        self.latencies.append(msg.data)

    def test_tunnel_pubsub(self):
        start = time.time()
        while time.time() - start < 12.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertGreaterEqual(self.subscriber_count, 5)

        if len(self.count_timestamps) >= 2:
            warmup = min(10, len(self.count_timestamps) // 5)
            steady = self.count_timestamps[warmup:]
            if len(steady) >= 2:
                fps = (len(steady) - 1) / (steady[-1] - steady[0])
                print(f'\n--- FPS: {fps:.1f}, frames: {self.subscriber_count} ---')

        if self.latencies:
            warmup = min(10, len(self.latencies) // 5)
            steady = self.latencies[warmup:]
            if steady:
                print(f'  E2E latency: min={min(steady):.2f} '
                      f'mean={sum(steady)/len(steady):.2f} '
                      f'max={max(steady):.2f} ms')


@launch_testing.post_shutdown_test()
class TestTunnelDemoShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -2, -6, -15])
