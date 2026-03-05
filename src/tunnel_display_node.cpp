// Copyright 2026 NVIDIA Corporation
// Licensed under the Apache License, Version 2.0

// Tunnel display node -- subscribes to tunnel frames and renders them.
//
// In OpenGL mode (default), frames are displayed in a GLFW window via
// CUDA-OpenGL interop: the received tensor is copied into a PBO that is
// mapped as a CUDA resource, avoiding any GPU-to-CPU round-trip.
//
// In headless mode (parameter headless:=true), the node skips rendering
// but still measures and publishes FPS and end-to-end latency for
// benchmarking.
//
// The window and GL resources are dynamically resized when the incoming
// image dimensions change.

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "std_msgs/msg/float64.hpp"
#include "torch_buffer/torch_buffer.hpp"

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
// X11 headers define macros that collide with C++ identifiers
#ifdef None
#undef None
#endif
#ifdef Bool
#undef Bool
#endif
#ifdef Status
#undef Status
#endif
#include <GLFW/glfw3.h>

static constexpr int DEFAULT_WIDTH  = 1280;
static constexpr int DEFAULT_HEIGHT = 720;

// OpenGL 3.3 core-profile function pointers loaded at runtime via glX
#define DECL_GL(ret, name, ...) static ret (APIENTRY *name)(__VA_ARGS__) = nullptr;
DECL_GL(GLuint, pglCreateShader,  GLenum)
DECL_GL(void,   pglShaderSource,  GLuint, GLsizei, const GLchar**, const GLint*)
DECL_GL(void,   pglCompileShader, GLuint)
DECL_GL(GLuint, pglCreateProgram)
DECL_GL(void,   pglAttachShader,  GLuint, GLuint)
DECL_GL(void,   pglLinkProgram,   GLuint)
DECL_GL(void,   pglDeleteShader,  GLuint)
DECL_GL(void,   pglDeleteProgram, GLuint)
DECL_GL(void,   pglUseProgram,    GLuint)
DECL_GL(void,   pglGenBuffers,    GLsizei, GLuint*)
DECL_GL(void,   pglDeleteBuffers, GLsizei, const GLuint*)
DECL_GL(void,   pglBindBuffer,    GLenum, GLuint)
DECL_GL(void,   pglBufferData,    GLenum, GLsizeiptr, const void*, GLenum)
DECL_GL(void,   pglGenVertexArrays,    GLsizei, GLuint*)
DECL_GL(void,   pglDeleteVertexArrays, GLsizei, const GLuint*)
DECL_GL(void,   pglBindVertexArray,    GLuint)
DECL_GL(void,   pglVertexAttribPointer, GLuint, GLint, GLenum, GLboolean, GLsizei, const void*)
DECL_GL(void,   pglEnableVertexAttribArray, GLuint)
#undef DECL_GL

#define LOAD_GL(name) *(void**)(&p##name) = (void*)glXGetProcAddress((const GLubyte*)#name)
static void loadGLFunctions()
{
  LOAD_GL(glCreateShader);  LOAD_GL(glShaderSource);  LOAD_GL(glCompileShader);
  LOAD_GL(glCreateProgram); LOAD_GL(glAttachShader);  LOAD_GL(glLinkProgram);
  LOAD_GL(glDeleteShader);  LOAD_GL(glDeleteProgram); LOAD_GL(glUseProgram);
  LOAD_GL(glGenBuffers);    LOAD_GL(glDeleteBuffers); LOAD_GL(glBindBuffer);
  LOAD_GL(glBufferData);
  LOAD_GL(glGenVertexArrays);    LOAD_GL(glDeleteVertexArrays);
  LOAD_GL(glBindVertexArray);    LOAD_GL(glVertexAttribPointer);
  LOAD_GL(glEnableVertexAttribArray);
}
#undef LOAD_GL

// Minimal fullscreen-quad shaders for texture display
static const char* vs_src = "#version 330 core\n"
  "layout(location=0) in vec2 pos; out vec2 uv;"
  "void main() { gl_Position = vec4(pos, 0, 1); uv = pos*0.5+0.5; }";
static const char* fs_src = "#version 330 core\n"
  "uniform sampler2D tex; in vec2 uv; out vec4 fragColor;"
  "void main() { fragColor = texture(tex, uv); }";

static GLuint make_program()
{
  GLuint vs = pglCreateShader(GL_VERTEX_SHADER);
  pglShaderSource(vs, 1, &vs_src, nullptr);
  pglCompileShader(vs);
  GLuint fs = pglCreateShader(GL_FRAGMENT_SHADER);
  pglShaderSource(fs, 1, &fs_src, nullptr);
  pglCompileShader(fs);
  GLuint prog = pglCreateProgram();
  pglAttachShader(prog, vs);
  pglAttachShader(prog, fs);
  pglLinkProgram(prog);
  pglDeleteShader(vs);
  pglDeleteShader(fs);
  return prog;
}

class TunnelDisplay : public rclcpp::Node
{
public:
  explicit TunnelDisplay(const rclcpp::NodeOptions & options)
  : Node("tunnel_display", options),
    frame_count_(0),
    total_count_(0),
    fps_timer_(std::chrono::steady_clock::now()),
    img_width_(DEFAULT_WIDTH), img_height_(DEFAULT_HEIGHT),
    win_(nullptr), pbo_(0), gl_tex_(0), vao_(0), vbo_(0), prog_(0),
    cuda_pbo_(nullptr), gl_ready_(false), headless_(false)
  {
    this->declare_parameter<bool>("headless", false);
    headless_ = this->get_parameter("headless").as_bool();

    if (!headless_) {
      cudaSetDevice(0);
      gl_ready_ = init_gl();
      if (!gl_ready_) {
        RCLCPP_WARN(this->get_logger(),
          "OpenGL init failed, falling back to headless mode");
        headless_ = true;
      }
    }

    auto qos = rclcpp::QoS(1).best_effort();
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "tunnel_image", qos,
      std::bind(&TunnelDisplay::image_callback, this, std::placeholders::_1));

    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>("subscriber_count", 10);
    latency_publisher_ = this->create_publisher<std_msgs::msg::Float64>("latency_ms", 10);

    if (!headless_) {
      glfw_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(4),
        std::bind(&TunnelDisplay::pump_glfw, this));
    }

    RCLCPP_INFO(this->get_logger(), "Tunnel display started (%dx%d, %s)",
      img_width_, img_height_, headless_ ? "headless" : "OpenGL + CUDA interop");
  }

  ~TunnelDisplay() override
  {
    if (cuda_pbo_) cudaGraphicsUnregisterResource(cuda_pbo_);
    if (gl_ready_) {
      pglDeleteBuffers(1, &pbo_);
      glDeleteTextures(1, &gl_tex_);
      pglDeleteProgram(prog_);
      pglDeleteVertexArrays(1, &vao_);
      pglDeleteBuffers(1, &vbo_);
    }
    if (win_) glfwDestroyWindow(win_);
    glfwTerminate();
  }

private:
  bool init_gl()
  {
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    win_ = glfwCreateWindow(img_width_, img_height_, "Tunnel Display", nullptr, nullptr);
    if (!win_) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(win_);
    glfwSwapInterval(0);
    loadGLFunctions();

    setup_gl_resources(img_width_, img_height_);

    float quad[] = { -1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1 };
    pglGenVertexArrays(1, &vao_);
    pglGenBuffers(1, &vbo_);
    pglBindVertexArray(vao_);
    pglBindBuffer(GL_ARRAY_BUFFER, vbo_);
    pglBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    pglVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    pglEnableVertexAttribArray(0);
    pglBindVertexArray(0);

    prog_ = make_program();
    return true;
  }

  // (Re)allocate PBO, CUDA-GL registration, and texture for the given size.
  void setup_gl_resources(int w, int h)
  {
    if (cuda_pbo_) {
      cudaGraphicsUnregisterResource(cuda_pbo_);
      cuda_pbo_ = nullptr;
    }
    if (pbo_) { pglDeleteBuffers(1, &pbo_); pbo_ = 0; }
    if (gl_tex_) { glDeleteTextures(1, &gl_tex_); gl_tex_ = 0; }

    pglGenBuffers(1, &pbo_);
    pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    pglBufferData(GL_PIXEL_UNPACK_BUFFER,
      static_cast<GLsizeiptr>(w) * h * 3, nullptr, GL_STREAM_DRAW);
    pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_, pbo_, cudaGraphicsMapFlagsWriteDiscard);

    glGenTextures(1, &gl_tex_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void pump_glfw()
  {
    if (!win_) return;
    glfwPollEvents();
    if (glfwWindowShouldClose(win_)) rclcpp::shutdown();
  }

  // Copy tensor data into the PBO via CUDA, then blit to the GL texture.
  void display_frame(const at::Tensor & tensor, int w, int h)
  {
    const size_t frame_bytes = static_cast<size_t>(w) * h * 3;
    cudaGraphicsMapResources(1, &cuda_pbo_, 0);
    void* d_pbo = nullptr;
    size_t sz = 0;
    cudaGraphicsResourceGetMappedPointer(&d_pbo, &sz, cuda_pbo_);
    auto kind = tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(d_pbo, tensor.data_ptr(), frame_bytes, kind);
    cudaGraphicsUnmapResources(1, &cuda_pbo_, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                    GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    pglUseProgram(prog_);
    pglBindVertexArray(vao_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    pglBindVertexArray(0);

    glfwSwapBuffers(win_);
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

    // Resize GL resources if the publisher changed resolution
    if (gl_ready_ && (w != img_width_ || h != img_height_)) {
      img_width_ = w;
      img_height_ = h;
      glfwSetWindowSize(win_, w, h);
      glViewport(0, 0, w, h);
      setup_gl_resources(w, h);
      RCLCPP_INFO(this->get_logger(), "Resized to %dx%d", w, h);
    }

    auto t0 = std::chrono::steady_clock::now();
    const rcl_buffer::Buffer<uint8_t> & data = msg->data;
    at::Tensor tensor = torch_buffer_backend::from_buffer(data);
    auto t1 = std::chrono::steady_clock::now();

    if (gl_ready_) {
      display_frame(tensor, w, h);
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
        gl_ready_ ? "GPU direct" : "headless");
      if (win_) {
        char title[80];
        snprintf(title, sizeof(title), "Tunnel Display -- %.1f fps", fps);
        glfwSetWindowTitle(win_, title);
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
  rclcpp::TimerBase::SharedPtr glfw_timer_;
  int frame_count_;
  uint32_t total_count_;
  std::chrono::steady_clock::time_point fps_timer_;
  std::chrono::steady_clock::time_point last_cb_end_{};
  double from_buf_sum_us_{0}, display_sum_us_{0}, publish_sum_us_{0};
  double total_cb_sum_us_{0}, gap_sum_us_{0};

  int img_width_, img_height_;
  GLFWwindow* win_;
  GLuint pbo_, gl_tex_, vao_, vbo_, prog_;
  cudaGraphicsResource_t cuda_pbo_;
  bool gl_ready_;
  bool headless_;
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
