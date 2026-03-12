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
#include "torch_buffer/torch_buffer.hpp"
#include "display.h"
#include <SDL.h>

class DisplayNode : public rclcpp::Node
{
public:
  explicit DisplayNode(const rclcpp::NodeOptions & options)
  : Node("display", options),
    frame_count_(0),
    fps_timer_(std::chrono::steady_clock::now()),
    headless_(false)
  {
    this->declare_parameter<bool>("headless", false);
    this->declare_parameter<bool>("use_cuda", true);
    this->declare_parameter<std::string>("record_path", "");
    headless_ = this->get_parameter("headless").as_bool();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    record_path_ = this->get_parameter("record_path").as_string();

    auto qos = rclcpp::QoS(1).reliable();
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "image", qos,
      std::bind(&DisplayNode::image_callback, this, std::placeholders::_1));

    if (!headless_) {
      event_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(4),
        std::bind(&DisplayNode::pump_events, this));
    }

    RCLCPP_INFO(this->get_logger(), "Display started (%s, waiting for first frame)",
      headless_ ? "headless" : (use_cuda_ ? "CUDA-GL interop" : "OpenGL"));
  }

  ~DisplayNode() override
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

  void report_fps(double latency_ms)
  {
    frame_count_++;
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - fps_timer_).count();
    if (elapsed < 1.0f) return;

    float fps = frame_count_ / elapsed;
    RCLCPP_INFO(this->get_logger(), "Display: %.1f fps | latency: %.2f ms | %s",
      fps, latency_ms,
      headless_ ? "headless" : (use_cuda_ ? "cuda" : "cpu"));
    if (display_ && display_->window()) {
      char title[128];
      snprintf(title, sizeof(title), "Display -- %.1f fps | %dx%d (%.1f MB)",
        fps, img_width_, img_height_, img_width_ * img_height_ * 4 / 1e6);
      SDL_SetWindowTitle(display_->window(), title);
    }
    frame_count_ = 0;
    fps_timer_ = now;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    ensure_display(static_cast<int>(msg->width), static_cast<int>(msg->height));

    auto guard = torch_buffer_backend::set_stream();
    int w = img_width_, h = img_height_;
    const rosidl::Buffer<uint8_t> & data = msg->data;
    at::Tensor frame = (data.get_backend_type() == "torch")
      ? torch_buffer_backend::from_buffer(data).reshape({h, w, 4})
      : torch::from_blob(const_cast<uint8_t *>(data.data()), {h, w, 4}, torch::kByte);

    if (!headless_ && display_)
      display_->present(frame);

    if (!record_path_.empty())
      record_frame(frame, img_width_, img_height_);

    double latency_ms = (this->now() - msg->header.stamp).seconds() * 1000.0;
    report_fps(latency_ms);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr event_timer_;
  int frame_count_;
  std::chrono::steady_clock::time_point fps_timer_;

  std::string record_path_;
  FILE * ffmpeg_pipe_{nullptr};
  std::vector<uint8_t> record_buf_;
  std::chrono::steady_clock::time_point last_record_time_{};

  int img_width_{0}, img_height_{0};
  bool use_cuda_;
  bool headless_;
  std::unique_ptr<FrameDisplay> display_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DisplayNode)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DisplayNode>(rclcpp::NodeOptions());
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
