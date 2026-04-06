// Copyright 2026 NVIDIA Corporation
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
#include "torch_buffer/torch_buffer_api.hpp"
#include "display.h"
#include "font.h"
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
    this->declare_parameter<bool>("borderless", false);
    this->declare_parameter<std::string>("record_path", "");
    this->declare_parameter<int>("window_x", -1);
    this->declare_parameter<int>("window_y", -1);
    this->declare_parameter<int>("max_window_width", 1920);
    this->declare_parameter<int>("max_window_height", 1080);
    headless_ = this->get_parameter("headless").as_bool();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    borderless_ = this->get_parameter("borderless").as_bool();
    record_path_ = this->get_parameter("record_path").as_string();
    win_x_ = static_cast<int>(this->get_parameter("window_x").as_int());
    win_y_ = static_cast<int>(this->get_parameter("window_y").as_int());
    max_win_w_ = static_cast<int>(this->get_parameter("max_window_width").as_int());
    max_win_h_ = static_cast<int>(this->get_parameter("max_window_height").as_int());

    auto qos = rclcpp::QoS(1).reliable();

    rclcpp::SubscriptionOptions sub_opts;
    if (use_cuda_) sub_opts.acceptable_buffer_backends = "any";
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "image", qos,
      std::bind(&DisplayNode::image_callback, this, std::placeholders::_1),
      sub_opts);

    if (!headless_) {
      event_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(4),
        std::bind(&DisplayNode::pump_events, this));
      label_ = make_text_bitmap(use_cuda_ ? "CUDA" : "CPU", 2, torch::kCPU);
    }

    RCLCPP_INFO(this->get_logger(), "Display started (%s%s, waiting for first frame)",
      headless_ ? "headless" : (use_cuda_ ? "CUDA-GL interop" : "OpenGL"),
      record_path_.empty() ? "" : ", recording");
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
    if (!display_->init(w, h, headless_, use_cuda_, false,
                        max_win_w_, max_win_h_, win_x_, win_y_, borderless_)) {
      RCLCPP_WARN(this->get_logger(), "Display init failed, falling back to headless");
      headless_ = true;
    }

    RCLCPP_INFO(this->get_logger(), "Display initialized: %dx%d (%s, window %dx%d)",
      w, h, headless_ ? "headless" : (use_cuda_ ? "CUDA-GL interop" : "OpenGL"),
      display_->win_width(), display_->win_height());
  }

  void pump_events()
  {
    if (display_ && !display_->poll_events()) rclcpp::shutdown();
  }

  static void stamp_label(at::Tensor & frame, const at::Tensor & label) {
    int lh = label.size(0), lw = label.size(1);
    int fh = frame.size(0), fw = frame.size(1);
    if (lh > fh || lw > fw) return;
    int y0 = 10, x0 = 16;
    if (y0 + lh > fh) return;
    auto roi = frame.index({
      torch::indexing::Slice(y0, y0 + lh),
      torch::indexing::Slice(x0, x0 + lw),
      torch::indexing::Slice(0, 3)
    });
    auto alpha = label.unsqueeze(2);
    if (frame.is_cuda()) alpha = alpha.to(frame.device());
    roi.copy_(roi * (1.0f - alpha));
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

    if (!headless_ && display_) {
      auto labeled = frame.clone();
      stamp_label(labeled, label_);
      display_->present(labeled);
    }

    if (!record_path_.empty())
      record_frame(frame, img_width_, img_height_);

    report_fps();
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

  void report_fps()
  {
    frame_count_++;
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - fps_timer_).count();
    if (elapsed < 1.0f) return;

    float fps = frame_count_ / elapsed;
    const char * tag = headless_ ? "headless" : (use_cuda_ ? "cuda" : "cpu");
    RCLCPP_INFO(this->get_logger(), "Display: %.1f fps | %s", fps, tag);

    if (!headless_) {
      char hud[64];
      snprintf(hud, sizeof(hud), "%s | %.0f FPS",
        use_cuda_ ? "CUDA" : "CPU", fps);
      label_ = make_text_bitmap(hud, 2, torch::kCPU);
    }

    frame_count_ = 0;
    fps_timer_ = now;
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr event_timer_;
  int frame_count_{0};
  std::chrono::steady_clock::time_point fps_timer_;

  std::string record_path_;
  FILE * ffmpeg_pipe_{nullptr};
  std::vector<uint8_t> record_buf_;
  std::chrono::steady_clock::time_point last_record_time_{};

  int img_width_{0}, img_height_{0};
  int win_x_, win_y_;
  int max_win_w_, max_win_h_;
  bool use_cuda_;
  bool headless_;
  bool borderless_;
  std::unique_ptr<FrameDisplay> display_;
  at::Tensor label_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DisplayNode)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DisplayNode>(rclcpp::NodeOptions());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
