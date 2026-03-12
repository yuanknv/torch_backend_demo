// Copyright 2024 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "std_msgs/msg/float64.hpp"
#include "torch_buffer/torch_buffer.hpp"
#include "display.h"
#include <SDL.h>

class TunnelDisplay : public rclcpp::Node
{
public:
  explicit TunnelDisplay(const rclcpp::NodeOptions & options)
  : Node("tunnel_display", options),
    frame_count_(0),
    total_count_(0),
    fps_timer_(std::chrono::steady_clock::now()),
    headless_(false)
  {
    this->declare_parameter<bool>("headless", false);
    this->declare_parameter<bool>("use_cuda", true);
    this->declare_parameter<std::string>("record_path", "");
    headless_ = this->get_parameter("headless").as_bool();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    record_path_ = this->get_parameter("record_path").as_string();

    auto qos = rclcpp::QoS(1).best_effort();
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "tunnel_image", qos,
      std::bind(&TunnelDisplay::image_callback, this, std::placeholders::_1));

    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>("subscriber_count", 10);
    latency_publisher_ = this->create_publisher<std_msgs::msg::Float64>("latency_ms", 10);

    if (!headless_) {
      event_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(4),
        std::bind(&TunnelDisplay::pump_events, this));
    }

    RCLCPP_INFO(this->get_logger(), "Display started (%s, waiting for first frame)",
      headless_ ? "headless" : (use_cuda_ ? "CUDA-GL interop" : "OpenGL"));
  }

  ~TunnelDisplay() override
  {
    if (ffmpeg_pipe_) {
      pclose(ffmpeg_pipe_);
      RCLCPP_INFO(this->get_logger(), "Video saved to %s", record_path_.c_str());
    }
  }

private:
  void ensure_display(int w, int h)
  {
    if (display_ && img_width_ == w && img_height_ == h) return;
    img_width_ = w;
    img_height_ = h;
    display_ = std::make_unique<FrameDisplay>();
    if (!display_->init(w, h, headless_, use_cuda_, false, 1920, 1080)) {
      RCLCPP_WARN(this->get_logger(), "Display init failed, falling back to headless");
      headless_ = true;
    }
    RCLCPP_INFO(this->get_logger(), "Display initialized: %dx%d (%s)",
      w, h, headless_ ? "headless" : (use_cuda_ ? "CUDA-GL interop" : "OpenGL"));
  }

  void pump_events()
  {
    if (display_ && !display_->poll_events()) rclcpp::shutdown();
  }

  void record_frame(const at::Tensor & tensor, int w, int h)
  {
    auto now = std::chrono::steady_clock::now();
    if (last_record_time_.time_since_epoch().count() > 0) {
      double elapsed_ms = std::chrono::duration<double, std::milli>(
        now - last_record_time_).count();
      if (elapsed_ms < 16.0) return;
    }
    last_record_time_ = now;

    if (!ffmpeg_pipe_) {
      std::string cmd =
        "ffmpeg -y -use_wallclock_as_timestamps 1"
        " -f rawvideo -pixel_format bgra"
        " -video_size " + std::to_string(w) + "x" + std::to_string(h) +
        " -i pipe:0"
        " -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -r 60"
        " " + record_path_ + " 2>/dev/null";
      ffmpeg_pipe_ = popen(cmd.c_str(), "w");
      if (!ffmpeg_pipe_) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open ffmpeg pipe");
        record_path_.clear();
        return;
      }
      record_buf_.resize(static_cast<size_t>(w) * h * 4);
      RCLCPP_INFO(this->get_logger(), "Recording started: %s (%dx%d @ 60fps)",
        record_path_.c_str(), w, h);
    }

    size_t frame_bytes = static_cast<size_t>(w) * h * 4;
    if (tensor.is_cuda()) {
      cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
      cudaMemcpyAsync(record_buf_.data(), tensor.data_ptr(), frame_bytes,
        cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    } else {
      std::memcpy(record_buf_.data(), tensor.data_ptr(), frame_bytes);
    }
    fwrite(record_buf_.data(), 1, frame_bytes, ffmpeg_pipe_);
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    auto cb_start = std::chrono::steady_clock::now();

    if (last_cb_end_.time_since_epoch().count() > 0) {
      double gap_us = std::chrono::duration<double, std::micro>(
        cb_start - last_cb_end_).count();
      gap_sum_us_ += gap_us;
    }

    int w = static_cast<int>(msg->width);
    int h = static_cast<int>(msg->height);

    ensure_display(w, h);

    auto guard = torch_buffer_backend::set_stream();
    auto t0 = std::chrono::steady_clock::now();
    const rosidl::Buffer<uint8_t> & data = msg->data;
    at::Tensor frame;
    if (data.get_backend_type() == "torch") {
      frame = torch_buffer_backend::from_buffer(data).reshape({h, w, 4});
    } else {
      frame = torch::from_blob(
        const_cast<uint8_t *>(data.data()), {h, w, 4}, torch::kByte);
    }
    auto t1 = std::chrono::steady_clock::now();

    if (!headless_ && display_) {
      display_->present(frame);
    }

    if (!record_path_.empty()) {
      record_frame(frame, w, h);
    }
    auto t2 = std::chrono::steady_clock::now();

    total_count_++;
    std_msgs::msg::UInt32 count_msg;
    count_msg.data = total_count_;
    count_publisher_->publish(count_msg);
    auto t3 = std::chrono::steady_clock::now();

    double latency_ms = (this->now() - msg->header.stamp).seconds() * 1000.0;

    std_msgs::msg::Float64 lat_msg;
    lat_msg.data = latency_ms;
    latency_publisher_->publish(lat_msg);

    double from_buf_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double display_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double publish_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    double total_cb_us = std::chrono::duration<double, std::micro>(t3 - cb_start).count();

    from_buf_sum_us_ += from_buf_us;
    display_sum_us_ += display_us;
    publish_sum_us_ += publish_us;
    total_cb_sum_us_ += total_cb_us;

    frame_count_++;
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - fps_timer_).count();
    if (elapsed >= 1.0f) {
      float fps = frame_count_ / elapsed;
      double n = frame_count_;
      RCLCPP_INFO(this->get_logger(),
        "Display: %.1f fps | latency: %.2f ms | cb: %.0f us "
        "(from_buf: %.0f, display: %.0f, pub: %.0f, gap: %.0f) | %s",
        fps, latency_ms,
        total_cb_sum_us_ / n,
        from_buf_sum_us_ / n, display_sum_us_ / n, publish_sum_us_ / n,
        gap_sum_us_ / n,
        headless_ ? "headless" : (use_cuda_ ? "cuda" : "cpu"));
      if (display_ && display_->window()) {
        char title[128];
        double mb = img_width_ * img_height_ * 4 / 1e6;
        snprintf(title, sizeof(title),
          "Display -- %.1f fps | %dx%d (%.1f MB)",
          fps, img_width_, img_height_, mb);
        SDL_SetWindowTitle(display_->window(), title);
      }
      frame_count_ = 0;
      fps_timer_ = now;
      from_buf_sum_us_ = 0; display_sum_us_ = 0;
      publish_sum_us_ = 0; total_cb_sum_us_ = 0; gap_sum_us_ = 0;
    }

    last_cb_end_ = std::chrono::steady_clock::now();
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_publisher_;
  rclcpp::TimerBase::SharedPtr event_timer_;
  int frame_count_;
  uint32_t total_count_;
  std::chrono::steady_clock::time_point fps_timer_;
  std::chrono::steady_clock::time_point last_cb_end_{};
  double from_buf_sum_us_{0}, display_sum_us_{0}, publish_sum_us_{0};
  double total_cb_sum_us_{0}, gap_sum_us_{0};

  std::string record_path_;
  FILE * ffmpeg_pipe_{nullptr};
  std::vector<uint8_t> record_buf_;
  std::chrono::steady_clock::time_point last_record_time_{};

  int img_width_{0}, img_height_{0};
  bool use_cuda_;
  bool headless_;
  std::unique_ptr<FrameDisplay> display_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(TunnelDisplay)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TunnelDisplay>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
