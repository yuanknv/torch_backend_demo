# tunnel_demo

A ROS 2 demo that renders an animated tunnel effect on the GPU with a CUDA kernel, publishes frames via the `torch_buffer_backend` zero-copy transport, and displays them in a GLFW window using CUDA-OpenGL interop.

The demo exercises the full buffer-aware pub/sub pipeline:

1. **tunnel_renderer_node** -- renders RGB frames on the GPU and publishes them as `sensor_msgs/msg/Image` using either CUDA IPC (zero-copy) or CPU (memcpy) transport.
2. **tunnel_display_node** -- subscribes, displays frames in an OpenGL window (or runs headless), and publishes FPS / end-to-end latency metrics.

## Dependencies

- [rcl_buffer_ws](https://github.com/yuankunz/rcl_buffer_ws) workspace with `pixi`
- `cuda_buffer_backend` and `torch_buffer_backend` packages (cloned via `vcs import`)
- CUDA toolkit, libtorch, GLFW, OpenGL

## Build

From the workspace root:

```bash
pixi run build tunnel_demo
```

## Run

```bash
pixi run bash src/tunnel_demo/launch/tunnel_demo.sh
```

Options:

```
--width WIDTH       Image width (default: 1920)
--height HEIGHT     Image height (default: 1080)
--backend BACKEND   cuda or cpu (default: cuda)
--rate RATE_MS      Publish timer period in ms (default: 1)
```

## Test

```bash
pixi run test tunnel_demo
```

The launch test spins up both nodes in headless mode for 12 seconds and asserts that frames are received with measured FPS and end-to-end latency.

Pass custom resolution/backend via launch args:

```bash
pixi run test tunnel_demo --tests_to_run test_tunnel_demo_launch
```

## Benchmark results

Measured on a single machine (inter-process, rmw_zenoh_cpp):

| Resolution | Image Size | Transport | FPS | Latency mean |
|---|---|---|---:|---:|
| 1280x720 | 2.8 MB | CUDA | 207 | 6.2 ms |
| 1280x720 | 2.8 MB | CPU | 190 | 30.4 ms |
| 1920x1080 | 6.2 MB | CUDA | 208 | 6.6 ms |
| 1920x1080 | 6.2 MB | CPU | 71 | 41.0 ms |
| 2560x1440 | 11.1 MB | CUDA | 208 | 6.9 ms |
| 2560x1440 | 11.1 MB | CPU | 28 | 104 ms |

CUDA transport throughput is resolution-independent since only descriptors are exchanged -- no pixel data is copied.

## License

Apache-2.0
