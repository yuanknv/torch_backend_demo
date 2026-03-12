#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    renderer_node = Node(
        package='torch_backend_demo',
        executable='renderer_node',
        name='renderer',
        output='screen',
    )

    display_node = Node(
        package='torch_backend_demo',
        executable='display_node',
        name='display',
        output='screen',
    )

    rmw_zenohd = os.path.join(
        get_package_prefix('rmw_zenoh_cpp'), 'lib', 'rmw_zenoh_cpp', 'rmw_zenohd')
    zenoh_router = ExecuteProcess(
        cmd=[rmw_zenohd],
        name='zenoh_router',
        output='screen',
    )

    ws_root = os.environ.get('PIXI_PROJECT_ROOT', '')
    if not ws_root:
        ws_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    libtorch_lib = os.path.join(ws_root, '.pixi', 'envs', 'default', 'libtorch', 'lib')
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if libtorch_lib not in ld_path:
        ld_path = f'{libtorch_lib}:{ld_path}' if ld_path else libtorch_lib

    return LaunchDescription([
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
                    ]),
                ],
            ),
        ),
    ])
