#include "display.h"
#include <GL/glew.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <algorithm>
#include <iostream>
#include <cstdio>

#ifdef USE_CUDA_GL_INTEROP
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

// Shader sources
static const char* vert_src = R"(
#version 330 core
const vec2 pos[4] = vec2[](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));
const vec2 uv[4]  = vec2[](vec2(0,1), vec2(1,1), vec2(1,0), vec2(0,0));
out vec2 v_uv;
void main() { gl_Position = vec4(pos[gl_VertexID], 0, 1); v_uv = uv[gl_VertexID]; }
)";

static const char* frag_src = R"(
#version 330 core
uniform sampler2D tex;
in vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = texture(tex, v_uv); }
)";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
    }
    return s;
}

FrameDisplay::FrameDisplay() {}

FrameDisplay::~FrameDisplay() {
    if (shader_program_) glDeleteProgram(shader_program_);
    if (tex_) glDeleteTextures(1, &tex_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
#ifdef USE_CUDA_GL_INTEROP
    if (cuda_pbo_) cudaGraphicsUnregisterResource((cudaGraphicsResource*)cuda_pbo_);
    if (pbo_) glDeleteBuffers(1, &pbo_);
#endif
    if (sdl_texture_) SDL_DestroyTexture((SDL_Texture*)sdl_texture_);
    if (sdl_renderer_) SDL_DestroyRenderer((SDL_Renderer*)sdl_renderer_);
    if (gl_ctx_) SDL_GL_DeleteContext(gl_ctx_);
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

bool FrameDisplay::init(int width, int height, bool headless, bool use_cuda, bool fullscreen,
                        int max_win_w, int max_win_h) {
    W_ = width;
    H_ = height;

    int winW = W_, winH = H_;
    if (max_win_w > 0 && max_win_h > 0 && (W_ > max_win_w || H_ > max_win_h)) {
        float scale = std::min(static_cast<float>(max_win_w) / W_,
                               static_cast<float>(max_win_h) / H_);
        winW = static_cast<int>(W_ * scale);
        winH = static_cast<int>(H_ * scale);
    }

    if (headless) {
        mode_ = DisplayMode::Headless;
        std::cout << "Display mode: Headless" << std::endl;
        return true;
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        mode_ = DisplayMode::Headless;
        return true; // fall back to headless
    }

    // Try OpenGL first
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    Uint32 win_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
    if (fullscreen) win_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;

    window_ = SDL_CreateWindow("Robot Arm Demo",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        winW, winH, win_flags);

    if (!window_) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        // Try software fallback
        Uint32 sw_flags = SDL_WINDOW_SHOWN;
        if (fullscreen) sw_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
        window_ = SDL_CreateWindow("Robot Arm Demo",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            winW, winH, sw_flags);
        if (!window_) {
            std::cerr << "SDL software window also failed" << std::endl;
            mode_ = DisplayMode::Headless;
            return true;
        }
        // Software SDL path
        mode_ = DisplayMode::SoftwareSDL;
        sdl_renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
        sdl_texture_ = SDL_CreateTexture((SDL_Renderer*)sdl_renderer_,
            SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, W_, H_);
        std::cout << "Display mode: SDL Software" << std::endl;
        return true;
    }

    gl_ctx_ = SDL_GL_CreateContext(window_);
    if (!gl_ctx_) {
        std::cerr << "GL context creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window_);
        window_ = nullptr;
        mode_ = DisplayMode::Headless;
        return true;
    }

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
        mode_ = DisplayMode::Headless;
        return true;
    }

    SDL_GL_SetSwapInterval(1);

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

#ifdef USE_CUDA_GL_INTEROP
    if (use_cuda) {
        mode_ = DisplayMode::CudaGL;
        std::cout << "Display mode: CUDA-GL Interop" << std::endl;
    } else
#endif
    {
        mode_ = DisplayMode::IntelGL;
        std::cout << "Display mode: Direct GL Upload" << std::endl;
    }

    init_gl_resources();
    return true;
}

void FrameDisplay::init_gl_resources() {
    // Compile shaders
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vert_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vs);
    glAttachShader(shader_program_, fs);
    glLinkProgram(shader_program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok;
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(shader_program_, 512, nullptr, log);
        std::cerr << "Shader link error: " << log << std::endl;
    }

    // Empty VAO (vertex data is generated in shader)
    glGenVertexArrays(1, &vao_);

    // Immutable texture
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W_, H_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

#ifdef USE_CUDA_GL_INTEROP
    if (mode_ == DisplayMode::CudaGL) {
        glGenBuffers(1, &pbo_);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, W_ * H_ * 4, nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        cudaGraphicsResource* resource;
        cudaGraphicsGLRegisterBuffer(&resource, pbo_, cudaGraphicsMapFlagsWriteDiscard);
        cuda_pbo_ = resource;
    }
#endif
}

void FrameDisplay::present(const torch::Tensor& frame) {
    switch (mode_) {
        case DisplayMode::CudaGL:
        case DisplayMode::IntelGL:
            present_gl(frame);
            break;
        case DisplayMode::SoftwareSDL:
            present_sdl_software(frame);
            break;
        case DisplayMode::Headless:
            break;
    }
}

void FrameDisplay::present_gl(const torch::Tensor& frame) {
    glBindTexture(GL_TEXTURE_2D, tex_);

#ifdef USE_CUDA_GL_INTEROP
    if (mode_ == DisplayMode::CudaGL && frame.is_cuda()) {
        cudaGraphicsResource* resource = (cudaGraphicsResource*)cuda_pbo_;
        cudaGraphicsMapResources(1, &resource, 0);
        void* d_ptr;
        size_t sz;
        cudaGraphicsResourceGetMappedPointer(&d_ptr, &sz, resource);
        cudaMemcpy(d_ptr, frame.data_ptr(), W_ * H_ * 4, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &resource, 0);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W_, H_,
                        GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    } else
#endif
    {
        auto cpu_frame = frame.to(torch::kCPU).contiguous();
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W_, H_,
                        GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                        cpu_frame.data_ptr<uint8_t>());
    }

    draw_fullscreen_quad();
    SDL_GL_SwapWindow(window_);
}

void FrameDisplay::present_sdl_software(const torch::Tensor& frame) {
    // BGRA -> ARGB for SDL: reorder channels [B,G,R,A] -> [A,R,G,B]
    auto cpu = frame.to(torch::kCPU).contiguous();
    auto argb = torch::stack({
        cpu.select(2, 3),  // A
        cpu.select(2, 2),  // R
        cpu.select(2, 1),  // G
        cpu.select(2, 0)   // B
    }, 2).contiguous();

    SDL_UpdateTexture((SDL_Texture*)sdl_texture_, nullptr,
                      argb.data_ptr<uint8_t>(), W_ * 4);
    SDL_RenderCopy((SDL_Renderer*)sdl_renderer_,
                   (SDL_Texture*)sdl_texture_, nullptr, nullptr);
    SDL_RenderPresent((SDL_Renderer*)sdl_renderer_);
}

void FrameDisplay::draw_fullscreen_quad() {
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program_);
    glBindVertexArray(vao_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glUniform1i(glGetUniformLocation(shader_program_, "tex"), 0);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

bool FrameDisplay::poll_events() {
    if (mode_ == DisplayMode::Headless) return true;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) return false;
        if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) return false;
    }
    return true;
}

void FrameDisplay::save_ppm(const torch::Tensor& frame_bgra, const std::string& path) {
    auto cpu = frame_bgra.to(torch::kCPU).contiguous();
    int H = cpu.size(0), W = cpu.size(1);
    // BGRA -> RGB
    auto rgb = torch::stack({cpu.select(2,2), cpu.select(2,1), cpu.select(2,0)}, 2)
               .contiguous();
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(rgb.data_ptr<uint8_t>(), 1, W * H * 3, f);
    fclose(f);
}
