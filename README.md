# tunnel_demo

A ROS 2 demo that renders animated GPU scenes, publishes frames via the `torch_buffer_backend` zero-copy transport, and displays them in a GLFW window using CUDA-OpenGL interop.

The demo exercises the full buffer-aware pub/sub pipeline:

1. **tunnel_renderer_node** -- renders RGB frames on the GPU and publishes them as `sensor_msgs/msg/Image` using either CUDA IPC (zero-copy) or CPU (memcpy) transport.
2. **tunnel_display_node** -- subscribes, displays frames in an OpenGL window (or runs headless), and publishes FPS / end-to-end latency metrics.

Two render scenes are available:

- **moving_objects** (default) -- a raymarched SDF scene with orbiting spheres, a rotating torus, and a bouncing box above a checkerboard ground plane. GPU-heavy.
- **tunnel** -- a neon-ring tunnel effect with wall illumination, depth fog, and procedural textures. Medium GPU load.

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
--width WIDTH       Image width (default: 2560)
--height HEIGHT     Image height (default: 1440)
--backend BACKEND   cuda or cpu (default: cuda)
--rate RATE_MS      Publish timer period in ms (default: 1)
--scene SCENE       Render scene: moving_objects or tunnel (default: moving_objects)
--record PATH       Record video to MP4 file (requires ffmpeg)
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

Measured on a single machine (inter-process, rmw_zenoh_cpp, headless mode).

### Moving objects scene

| Resolution | Image Size | Transport | FPS | Latency mean |
|---|---|---|---:|---:|
| 1920x1080 | 6.2 MB | CUDA | 100.5 | 21.5 ms |
| 1920x1080 | 6.2 MB | CPU | 59.9 | 30.1 ms |
| 2560x1440 | 11.1 MB | CUDA | 67.0 | 22.3 ms |
| 2560x1440 | 11.1 MB | CPU | 27.2 | 109.7 ms |
| 6144x3160 | 58.2 MB | CUDA | 13.4 | 81.2 ms |
| 6144x3160 | 58.2 MB | CPU | 3.6 | 510.4 ms |

### Tunnel effect scene

| Resolution | Image Size | Transport | FPS | Latency mean |
|---|---|---|---:|---:|
| 1920x1080 | 6.2 MB | CUDA | 99.8 | 88.0 ms |
| 1920x1080 | 6.2 MB | CPU | 59.3 | 47.6 ms |
| 2560x1440 | 11.1 MB | CUDA | 97.7 | 86.9 ms |
| 2560x1440 | 11.1 MB | CPU | 25.2 | 116.2 ms |
| 6144x3160 | 58.2 MB | CUDA | 100.7 | 27.8 ms |
| 6144x3160 | 58.2 MB | CPU | 4.4 | 640.4 ms |

At high resolutions the FPS gap between CUDA and CPU transport widens significantly. The moving objects scene is GPU-bound (raymarching cost scales with resolution), while the tunnel scene is lighter so CUDA transport maintains ~100 FPS even at 6K. CPU transport drops to single digits at 6K due to the ~58 MB per-frame memcpy overhead.

## License

Apache-2.0
