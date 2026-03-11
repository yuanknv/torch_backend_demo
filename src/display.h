#pragma once
#include <torch/torch.h>
#include <string>

struct SDL_Window;
typedef void* SDL_GLContext;

enum class DisplayMode { CudaGL, IntelGL, SoftwareSDL, Headless };

class FrameDisplay {
public:
    FrameDisplay();
    ~FrameDisplay();

    bool init(int width, int height, bool headless, bool use_cuda, bool fullscreen = false);
    void present(const torch::Tensor& frame);
    bool poll_events(); // returns false if quit requested

    static void save_ppm(const torch::Tensor& frame_bgra, const std::string& path);

    DisplayMode mode() const { return mode_; }
    SDL_Window* window() const { return window_; }

private:
    void init_gl_resources();
    void present_gl(const torch::Tensor& frame);
    void present_sdl_software(const torch::Tensor& frame);
    void draw_fullscreen_quad();

    DisplayMode mode_ = DisplayMode::Headless;
    int W_ = 0, H_ = 0;

    SDL_Window* window_ = nullptr;
    SDL_GLContext gl_ctx_ = nullptr;

    // GL resources
    unsigned int tex_ = 0;
    unsigned int vao_ = 0;
    unsigned int shader_program_ = 0;

#ifdef USE_CUDA_GL_INTEROP
    unsigned int pbo_ = 0;
    void* cuda_pbo_ = nullptr;
#endif

    // SDL software fallback
    void* sdl_renderer_ = nullptr;
    void* sdl_texture_ = nullptr;
};
