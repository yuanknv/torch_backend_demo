// Copyright 2024 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

// Tunnel renderer node -- publishes animated tunnel frames as sensor_msgs/Image.
//
// The tunnel effect is rendered entirely via LibTorch tensor ops (no custom
// CUDA kernels). Every pixel is computed as a batched tensor operation over
// the full [H x W] grid. The only host-side loop is the ring iteration.
//
// Supports two transport modes selected via the 'use_cuda' parameter:
//   cuda: allocates a CUDA buffer via torch_buffer_backend and renders directly
//         into it.  The subscriber receives a zero-copy CUDA IPC handle.
//   cpu:  renders on the GPU, copies to host, and writes into a CPU buffer.

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "torch_buffer/torch_buffer.hpp"

static constexpr float RING_SPACING  = 4.0f;
static constexpr float RING_RADIUS   = 1.8f;
static constexpr float TUBE_WIDTH    = 0.08f;
static constexpr float CAM_SPEED     = 5.0f;
static constexpr float COLOR_SCROLL  = 0.35f;
static constexpr float HUE_STEP      = 0.12f;
static constexpr int   RINGS_VISIBLE = 16;
static constexpr float FOV_Y         = 1.05f;

struct RGB { float r, g, b; };
static RGB hsv2rgb(float h, float s, float v)
{
    h -= std::floor(h);
    float H = h * 6.0f;
    int   i = static_cast<int>(H);
    float f = H - i;
    float p = v * (1.f - s);
    float q = v * (1.f - s * f);
    float t = v * (1.f - s * (1.f - f));
    switch (i % 6) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default:return {v, p, q};
    }
}

struct RayGrid {
    torch::Tensor inv_rdz;
    torch::Tensor rdx_over_rdz, rdy_over_rdz;
    torch::Tensor vignette;
    int W = 0, H = 0;
};

static RayGrid buildRayGrid(int W, int H, torch::Device dev)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    float aspect = static_cast<float>(W) / H;
    float halfH  = std::tan(FOV_Y * 0.5f);
    auto px = torch::arange(W, opts).unsqueeze(0);
    auto py = torch::arange(H, opts).unsqueeze(1);
    auto u = ((px + 0.5f) / W - 0.5f) * 2.f * aspect * halfH;
    auto v = (0.5f - (py + 0.5f) / H) * 2.f * halfH;
    auto inv_len = torch::rsqrt(u * u + v * v + 1.f);
    auto rdz = inv_len;
    auto inv_rdz = 1.f / rdz;
    auto rdx_over_rdz = (u * inv_len) / rdz;
    auto rdy_over_rdz = (v * inv_len) / rdz;
    auto vignette = (1.f - 0.4f * (u * u + v * v) / (halfH * halfH * aspect)).clamp(0.f, 1.f);
    return {inv_rdz, rdx_over_rdz, rdy_over_rdz, vignette, W, H};
}

static at::Tensor renderTunnel(const RayGrid& rg, float time)
{
    auto dev  = rg.inv_rdz.device();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    // Single [H, W, 3] accumulator instead of 3 separate [H, W] tensors
    auto acc = torch::zeros({rg.H, rg.W, 3}, opts);

    const float camZ    = time * CAM_SPEED;
    const int   baseIdx = static_cast<int>(std::floor(camZ / RING_SPACING));
    const float inv_tw2 = 1.f / (TUBE_WIDTH * TUBE_WIDTH);

    for (int di = -2; di < RINGS_VISIBLE; ++di) {
        const int   ringIdx    = baseIdx + di;
        const float planeZ     = ringIdx * RING_SPACING - camZ;
        if (planeZ <= 0.f) continue;

        // t = planeZ / rdz = planeZ * inv_rdz; all valid since planeZ > 0
        auto t = rg.inv_rdz * planeZ;
        // Hit point: ix = rdx * t = (rdx/rdz) * planeZ, same for iy
        // r^2 = ix^2 + iy^2 = planeZ^2 * (rdx_over_rdz^2 + rdy_over_rdz^2)
        auto r2 = (rg.rdx_over_rdz * rg.rdx_over_rdz
                  + rg.rdy_over_rdz * rg.rdy_over_rdz) * (planeZ * planeZ);
        // (r - R)^2 = r2 - 2R*sqrt(r2) + R^2; approximate: use (sqrt(r2) - R)^2
        auto r = torch::sqrt(r2);
        auto dr2 = (r - RING_RADIUS).square_();
        // Combined glow: exp(-(dr/tw)^2 - t*0.07) in one exp call
        auto glow = torch::exp(-dr2 * inv_tw2 - t * 0.07f);

        float hue        = ringIdx * HUE_STEP + time * COLOR_SCROLL;
        float brightness = 0.75f + 0.25f * std::sin(ringIdx * 1.3f + time * 1.7f);
        RGB   rc         = hsv2rgb(hue, 1.f, brightness);

        // Build [1, 1, 3] color tensor, broadcast-multiply with [H, W] glow
        float rgb[3] = {rc.r, rc.g, rc.b};
        auto color = torch::from_blob(rgb, {1, 1, 3}, torch::kFloat32).to(dev);
        acc.add_(glow.unsqueeze(2) * color);
    }

    acc *= rg.vignette.unsqueeze(2);
    acc = acc / (1.f + acc);
    acc = torch::pow(acc, 1.f / 2.2f);
    return acc.mul_(255.f).clamp_(0.f, 255.f).to(torch::kUInt8);
}

class TunnelRenderer : public rclcpp::Node
{
public:
  explicit TunnelRenderer(const rclcpp::NodeOptions & options)
  : Node("tunnel_renderer", options),
    frame_count_(0),
    t0_(std::chrono::steady_clock::now()),
    fps_timer_(t0_),
    use_cuda_(true)
  {
    this->declare_parameter<int>("publish_rate_ms", 1);
    this->declare_parameter<bool>("use_cuda", true);
    this->declare_parameter<int>("image_width", 1920);
    this->declare_parameter<int>("image_height", 1080);
    int rate_ms = this->get_parameter("publish_rate_ms").as_int();
    if (rate_ms <= 0) rate_ms = 1;
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    width_ = this->get_parameter("image_width").as_int();
    height_ = this->get_parameter("image_height").as_int();

    torch::Device dev = use_cuda_ ? torch::kCUDA : torch::kCPU;
    ray_grid_ = buildRayGrid(width_, height_, dev);

    auto qos = rclcpp::QoS(1).best_effort();
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("tunnel_image", qos);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(rate_ms),
      std::bind(&TunnelRenderer::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
      "Tunnel renderer started (%dx%d, %.1f MB, timer=%dms, transport=%s)",
      width_, height_, width_ * height_ * 3 / 1e6,
      rate_ms, use_cuda_ ? "cuda" : "cpu");
  }

private:
  void timer_callback()
  {
    auto cb_start = std::chrono::steady_clock::now();
    if (last_cb_end_.time_since_epoch().count() > 0) {
      gap_sum_us_ += std::chrono::duration<double, std::micro>(
        cb_start - last_cb_end_).count();
    }

    auto guard = torch_buffer_backend::set_stream();
    c10::DeviceType transport = use_cuda_ ? c10::kCUDA : c10::kCPU;
    rclcpp::Time e2e_start = this->now();

    auto t_alloc = std::chrono::steady_clock::now();
    sensor_msgs::msg::Image msg =
      torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
        {height_, width_, 3}, torch::kByte, transport);
    auto t_alloc_end = std::chrono::steady_clock::now();

    msg.header.stamp = e2e_start;
    msg.header.frame_id = "tunnel";
    msg.height = height_;
    msg.width = width_;
    msg.encoding = "rgb8";
    msg.step = width_ * 3;
    msg.is_bigendian = 0;

    float t = std::chrono::duration<float>(
      std::chrono::steady_clock::now() - t0_).count();

    auto t_render = std::chrono::steady_clock::now();
    at::Tensor frame = renderTunnel(ray_grid_, t);

    at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
    if (use_cuda_) {
      output.copy_(frame);
    } else {
      at::Tensor cpu_frame = frame.cpu();
      torch_buffer_backend::to_buffer(cpu_frame, msg.data);
    }
    auto t_render_end = std::chrono::steady_clock::now();

    auto t_pub = std::chrono::steady_clock::now();
    publisher_->publish(msg);
    auto t_pub_end = std::chrono::steady_clock::now();

    double alloc_us = std::chrono::duration<double, std::micro>(t_alloc_end - t_alloc).count();
    double render_us = std::chrono::duration<double, std::micro>(t_render_end - t_render).count();
    double pub_us = std::chrono::duration<double, std::micro>(t_pub_end - t_pub).count();
    double total_us = std::chrono::duration<double, std::micro>(t_pub_end - cb_start).count();
    alloc_sum_us_ += alloc_us;
    render_sum_us_ += render_us;
    pub_sum_us_ += pub_us;
    total_sum_us_ += total_us;

    frame_count_++;
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - fps_timer_).count();
    if (elapsed >= 1.0f) {
      double n = frame_count_;
      RCLCPP_INFO(this->get_logger(),
        "Publishing: %.1f fps [%s] | cb: %.0f us (alloc: %.0f, render: %.0f, pub: %.0f, gap: %.0f)",
        frame_count_ / elapsed, use_cuda_ ? "cuda" : "cpu",
        total_sum_us_ / n, alloc_sum_us_ / n, render_sum_us_ / n,
        pub_sum_us_ / n, gap_sum_us_ / n);
      frame_count_ = 0;
      fps_timer_ = now;
      alloc_sum_us_ = 0; render_sum_us_ = 0; pub_sum_us_ = 0;
      total_sum_us_ = 0; gap_sum_us_ = 0;
    }
    last_cb_end_ = std::chrono::steady_clock::now();
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  int frame_count_;
  std::chrono::steady_clock::time_point t0_;
  std::chrono::steady_clock::time_point fps_timer_;
  bool use_cuda_;
  int width_, height_;
  RayGrid ray_grid_;
  std::chrono::steady_clock::time_point last_cb_end_{};
  double alloc_sum_us_{0}, render_sum_us_{0}, pub_sum_us_{0};
  double total_sum_us_{0}, gap_sum_us_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(TunnelRenderer)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TunnelRenderer>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
