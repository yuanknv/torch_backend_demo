#pragma once
#include <torch/torch.h>
#include <array>

class RobotArmRenderer {
public:
    RobotArmRenderer(int width, int height, torch::Device device);

    void update(float dt);
    torch::Tensor render_frame();

private:
    // Rendering layers
    torch::Tensor render_background();
    void render_table(torch::Tensor& frame);
    void render_cubes(torch::Tensor& frame);
    void render_arm(torch::Tensor& frame);
    void render_gripper(torch::Tensor& frame, float ex, float ey);

    // 3-link IK: solve for shoulder, elbow, and wrist world angle.
    // Link3 tilts slightly toward target for visible wrist articulation.
    bool solve_ik(float tx, float ty, float& shoulder, float& elbow, float& wrist_world);

    // Forward kinematics: returns joint positions for all 3 links
    void forward_kin(float shoulder, float elbow, float wrist_world,
                     float& elbow_x, float& elbow_y,
                     float& wrist_x, float& wrist_y,
                     float& ee_x, float& ee_y);

    int W_, H_;
    torch::Device device_;

    // Pre-computed coordinate grids [H,W] float32
    torch::Tensor x_, y_;
    // Pre-computed pencil textures [H,W] float32 (computed once in constructor)
    torch::Tensor wobble_, grain_;

    // Localized drawing: operate on bounding-box slices instead of full frame
    void draw_local_sketch_box(torch::Tensor& frame, float cx, float cy,
                                float hx, float hy, float lw, float darkness);
    void draw_local_sketch_box_rotated(torch::Tensor& frame, float cx, float cy,
                                        float hx, float hy, float angle,
                                        float lw, float darkness);
    void draw_local_sketch_circle(torch::Tensor& frame, float cx, float cy,
                                   float radius, float lw, float darkness);
    void draw_local_color_pencil_box(torch::Tensor& frame, float cx, float cy,
                                      float hx, float hy, float angle,
                                      float col_r, float col_g, float col_b,
                                      float lw, float outline_darkness);

    // Animation time
    float time_ = 0.0f;

    // ── Robot geometry (in pixel coords, origin top-left) ──
    float base_x_, base_y_;     // shoulder joint position (top of pedestal)
    float pedestal_top_;        // top of the table where pedestal sits
    float pedestal_h_;          // height of pedestal
    float link1_len_, link2_len_, link3_len_;

    // Current joint angles (radians, from +x CCW)
    float shoulder_angle_ = 0.0f;
    float elbow_angle_ = 0.0f;
    float wrist_angle_ = 0.0f;  // world-frame angle of link3

    // Home position angles
    float home_shoulder_, home_elbow_, home_wrist_;

    // Gripper geometry (all derived from cube_half_ at construction time)
    float gripper_opening_ = 1.0f;  // 0=closed (clamped), 1=open
    float gripper_target_ = 1.0f;
    static constexpr float GRIPPER_SPEED = 4.0f;

    float jaw_w_;          // horizontal thickness of each jaw
    float jaw_h_;          // vertical height of each jaw
    float open_offset_;    // jaw center offset from ee center when open
    float closed_offset_;  // jaw center offset when clamped (inner edge = cube edge)
    float grip_grab_offset_; // distance from ee_y to cube center_y when held

    // ── Cubes ──
    struct Cube {
        float x, y;          // center position
        float size;           // half-size
        float r, g, b;       // color (RGB)
        bool stacked = false; // currently in the stack
        bool held = false;    // currently held by gripper
        float vx = 0, vy = 0; // velocity for collapse physics
        float angle = 0;      // rotation angle
    };
    std::array<Cube, 3> cubes_;
    float cube_half_ = 0.0f;

    // Pre-rendered label bitmaps [h,w] float (one per cube, assigned per pick order)
    std::array<torch::Tensor, 3> label_bmps_;

    // Table surface
    float table_y_;  // top of table (y increases downward)

    // Stack position
    float stack_x_;

    // ── State machine ──
    enum class State {
        Idle,           // brief pause at home
        MoveToAbove,    // move above target cube
        LowerToGrab,    // lower to cube
        CloseGripper,   // close gripper
        LiftCube,       // lift cube up
        MoveToStack,    // move above stack position
        LowerToStack,   // lower cube to stack
        OpenGripper,    // release cube
        LiftFromStack,  // lift away from stack
        ReturnHome,     // go back to home
        Collapse,       // stack collapses
        Scatter,        // cubes fly to new positions
    };

    State state_ = State::Idle;
    float state_timer_ = 0.0f;
    int current_pick_ = 0;   // index into pick_order_ (0, 1, 2)
    int cubes_stacked_ = 0;
    std::array<int, 3> pick_order_;  // indices into cubes_, computed for collision-free stacking

    // Motion interpolation
    float move_start_shoulder_, move_start_elbow_, move_start_wrist_;
    float move_target_shoulder_, move_target_elbow_, move_target_wrist_;
    float move_duration_ = 0.0f;
    float move_elapsed_ = 0.0f;

    // Helpers
    void start_move_to(float tx, float ty, float duration);
    void randomize_cube_positions();
    void compute_scatter_targets();  // pre-compute next positions, launch cubes
    float smooth_step(float t);

    // Scatter targets: sorted x positions for the next cycle
    std::array<float, 3> scatter_target_x_ = {0, 0, 0};

    // Pseudo-random
    float rand_float(int seed);
    int rand_seed_ = 42;
};
