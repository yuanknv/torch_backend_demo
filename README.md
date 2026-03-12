# torch_backend_demo

A ROS 2 demo that renders an SDF-based pencil-sketch robot arm animation entirely on the GPU via LibTorch tensor ops, publishes BGRA frames as `sensor_msgs/msg/Image`, and displays them in an SDL2/OpenGL window with CUDA-GL interop.

Two transport modes are compared:

- **cuda** -- uses the `torch_buffer_backend` with CUDA IPC for zero-copy GPU-to-GPU frame transport between processes.
- **cpu** -- publishes raw `sensor_msgs/msg/Image` data (GPU render, then `cudaMemcpy` to host, serialised via standard ROS 2 middleware). No buffer backend is involved.

Both modes render on the GPU. The only difference is the transport path, making this a clean comparison of zero-copy CUDA IPC vs traditional CPU-serialised image transport.

Animation is frame-count driven (fixed dt = 1/60 s per frame), so low FPS results in slower but smooth playback rather than frame skipping.

The demo exercises the full buffer-aware pub/sub pipeline:

1. **renderer_node** -- renders BGRA frames on the GPU using LibTorch SDF operations and publishes them as `sensor_msgs/msg/Image` via either CUDA IPC or raw CPU transport.
2. **display_node** -- subscribes, displays frames in an SDL2/OpenGL window (with CUDA-GL interop for the CUDA path, or CPU texture upload for the raw path), and reports FPS / end-to-end latency.

## Dependencies

- [rcl_buffer_ws](https://github.com/yuankunz/rcl_buffer_ws) workspace with `pixi`
- `cuda_buffer_backend` and `torch_buffer_backend` packages (cloned via `vcs import`)
- CUDA toolkit, libtorch, SDL2, GLEW, OpenGL

## Build

From the workspace root:

```bash
pixi run build torch_backend_demo
```

## Run

```bash
pixi run bash src/torch_backend_demo/launch/demo.sh
```

Options:

```
--width WIDTH       Image width (default: 1920)
--height HEIGHT     Image height (default: 1080)
--backend BACKEND   cuda or cpu (default: cuda)
--rate RATE_MS      Publish timer period in ms (default: 4)
--record PATH       Record video to MP4 file (requires ffmpeg)
```

## Test

```bash
pixi run test torch_backend_demo
```

The launch test spins up both nodes in headless mode for 12 seconds and asserts that frames are received with measured FPS and end-to-end latency.

## Benchmark results

Measured on a single machine (inter-process, rmw_zenoh_cpp, headless mode):

- **GPU**: NVIDIA GeForce RTX 3090 (24 GB)
- **CPU**: Intel Core i9-10850K @ 3.60 GHz
- **RMW**: rmw_zenoh_cpp

To reproduce, build the workspace and run:

```bash
pixi run bash src/torch_backend_demo/launch/bench_all.sh
```

| Resolution | Image Size | Transport | FPS | E2E Latency | Speedup |
|---|---|---|---:|---:|---:|
| 1920x1080 | 7.9 MB | CUDA | 101.1 | 35.7 ms | 1.3x |
| 1920x1080 | 7.9 MB | CPU | 79.5 | 24.7 ms | -- |
| 2560x1440 | 14.1 MB | CUDA | 102.3 | 38.9 ms | 3.6x |
| 2560x1440 | 14.1 MB | CPU | 28.1 | 100.5 ms | -- |
| 3840x2160 | 31.6 MB | CUDA | 103.1 | 24.4 ms | 8.6x |
| 3840x2160 | 31.6 MB | CPU | 12.0 | 241.0 ms | -- |

The CUDA path maintains ~100 FPS across all resolutions because zero-copy IPC cost is nearly constant regardless of frame size. The CPU path must copy frames from GPU to host and serialise them through the middleware, so its throughput drops as image size grows. At 4K (31.6 MB/frame) the CUDA backend is ~8.6x faster than the raw CPU path.

## License

Apache-2.0
