// Copyright 2024 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "torch_buffer/torch_buffer.hpp"
#include "robot_arm.h"

class TunnelRenderer : public rclcpp::Node
{
public:
  explicit TunnelRenderer(const rclcpp::NodeOptions & options)
  : Node("tunnel_renderer", options),
    frame_count_(0),
    t0_(std::chrono::steady_clock::now()),
    fps_timer_(t0_)
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

    renderer_ = std::make_unique<RobotArmRenderer>(width_, height_, torch::kCUDA);

    auto qos = rclcpp::QoS(1).best_effort();
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("tunnel_image", qos);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(rate_ms),
      std::bind(&TunnelRenderer::timer_callback, this));

    RCLCPP_INFO(this->get_logger(),
      "Robot arm renderer started (%dx%d, %.1f MB, timer=%dms, transport=%s)",
      width_, height_, width_ * height_ * 4 / 1e6,
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
    rclcpp::Time e2e_start = this->now();

    constexpr float dt = 1.0f / 60.0f;

    auto t_alloc = std::chrono::steady_clock::now();
    sensor_msgs::msg::Image msg;
    if (use_cuda_) {
      msg = torch_buffer_backend::allocate_msg<sensor_msgs::msg::Image>(
        {height_, width_, 4}, torch::kByte, c10::kCUDA);
    }
    auto t_alloc_end = std::chrono::steady_clock::now();

    msg.header.stamp = e2e_start;
    msg.header.frame_id = "tunnel";
    msg.height = height_;
    msg.width = width_;
    msg.encoding = "bgra8";
    msg.step = width_ * 4;
    msg.is_bigendian = 0;

    auto t_render = std::chrono::steady_clock::now();
    renderer_->update(dt);
    at::Tensor frame = renderer_->render_frame();

    if (use_cuda_) {
      at::Tensor output = torch_buffer_backend::from_buffer(msg.data);
      output.copy_(frame);
    } else {
      at::Tensor cpu_frame = frame.cpu().contiguous();
      size_t nbytes = static_cast<size_t>(height_) * width_ * 4;
      msg.data.resize(nbytes);
      std::memcpy(msg.data.data(), cpu_frame.data_ptr(), nbytes);
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
  std::unique_ptr<RobotArmRenderer> renderer_;
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
