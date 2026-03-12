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
from sensor_msgs.msg import Image


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    width_arg = DeclareLaunchArgument('width', default_value='1920')
    height_arg = DeclareLaunchArgument('height', default_value='1080')
    use_cuda_arg = DeclareLaunchArgument('use_cuda', default_value='true')

    renderer_node = Node(
        package='torch_backend_demo',
        executable='renderer_node',
        name='renderer',
        output='screen',
        parameters=[{
            'image_width': LaunchConfiguration('width'),
            'image_height': LaunchConfiguration('height'),
            'use_cuda': LaunchConfiguration('use_cuda'),
        }],
    )

    display_node = Node(
        package='torch_backend_demo',
        executable='display_node',
        name='display',
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


class TestDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_torch_backend_demo')
        self.frame_count = 0
        self.frame_timestamps = []
        self.node.create_subscription(
            Image, 'image', self._image_cb,
            rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE))

    def tearDown(self):
        self.node.destroy_node()

    def _image_cb(self, msg):
        self.frame_count += 1
        self.frame_timestamps.append(time.time())

    def test_pubsub(self):
        start = time.time()
        while time.time() - start < 12.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertGreaterEqual(self.frame_count, 5)

        if len(self.frame_timestamps) >= 2:
            warmup = min(10, len(self.frame_timestamps) // 5)
            steady = self.frame_timestamps[warmup:]
            if len(steady) >= 2:
                fps = (len(steady) - 1) / (steady[-1] - steady[0])
                print(f'\n--- FPS: {fps:.1f}, frames: {self.frame_count} ---')


@launch_testing.post_shutdown_test()
class TestDemoShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -2, -6, -15])
