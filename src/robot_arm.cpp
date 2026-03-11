#include "robot_arm.h"
#include "font.h"
#include <cmath>
#include <algorithm>
#include <cstdint>

// ── Pencil sketch style: gray/black outlines on white ───────────────────
// Pencil darkness levels
static constexpr float PENCIL_DARK = 35.0f;    // dark strokes
static constexpr float PENCIL_MED  = 70.0f;    // medium strokes
static constexpr float PENCIL_LIGHT = 130.0f;  // light strokes

// ── Helpers ──────────────────────────────────────────────────────────────

static inline torch::Tensor sdf_box(const torch::Tensor& px, const torch::Tensor& py,
                                     float cx, float cy, float hx, float hy) {
    auto dx = (px - cx).abs() - hx;
    auto dy = (py - cy).abs() - hy;
    return torch::max(dx, dy);
}

static inline torch::Tensor sdf_box_rotated(const torch::Tensor& px, const torch::Tensor& py,
                                              float cx, float cy, float hx, float hy,
                                              float angle) {
    float ca = std::cos(angle), sa = std::sin(angle);
    auto dx = px - cx;
    auto dy = py - cy;
    auto rx = (dx * ca + dy * sa).abs() - hx;
    auto ry = (-dx * sa + dy * ca).abs() - hy;
    return torch::max(rx, ry);
}

// Hand-drawn wobble field (shared so parallel strokes stay coherent).
static inline torch::Tensor pencil_wobble(const torch::Tensor& x_, const torch::Tensor& y_) {
    return torch::sin(x_ * 0.05f + y_ * 0.07f) * 0.6f
         + torch::sin(x_ * 0.03f - y_ * 0.04f + 0.7f) * 0.4f;
}

// Pencil grain texture: high-frequency noise.
static inline torch::Tensor pencil_grain(const torch::Tensor& x_, const torch::Tensor& y_) {
    return torch::sin(x_ * 1.7f + y_ * 2.3f) *
           torch::cos(x_ * 3.1f - y_ * 1.1f) * 0.15f + 0.85f;
}

// Pencil stroke from pre-wobbled SDF (no additional wobble applied).
static inline torch::Tensor pencil_stroke_raw(const torch::Tensor& psdf, float lw,
                                               const torch::Tensor& grain) {
    auto core = torch::exp(-(psdf / lw).square());
    return core * grain;
}

// ── Localized draw methods (operate on bounding-box slices) ─────────────

using S = torch::indexing::Slice;

// Compute tight bounding box, return false if empty
static bool compute_bbox(float cx, float cy, float extent, float margin,
                          int W, int H, int& y0, int& y1, int& x0, int& x1) {
    x0 = std::max(0, (int)std::floor(cx - extent - margin));
    x1 = std::min(W, (int)std::ceil(cx + extent + margin));
    y0 = std::max(0, (int)std::floor(cy - extent - margin));
    y1 = std::min(H, (int)std::ceil(cy + extent + margin));
    return x0 < x1 && y0 < y1;
}

void RobotArmRenderer::draw_local_sketch_box(torch::Tensor& frame, float cx, float cy,
                                               float hx, float hy, float lw, float darkness) {
    float extent = std::max(hx, hy);
    float margin = lw * 4.0f + 3.0f;
    int y0, y1, x0, x1;
    if (!compute_bbox(cx, cy, extent, margin, W_, H_, y0, y1, x0, x1)) return;

    auto lx = x_.index({S(y0, y1), S(x0, x1)});
    auto ly = y_.index({S(y0, y1), S(x0, x1)});
    auto lw_ = wobble_.index({S(y0, y1), S(x0, x1)});
    auto lg = grain_.index({S(y0, y1), S(x0, x1)});

    auto sdf = sdf_box(lx, ly, cx, cy, hx, hy);
    auto psdf = sdf + lw_;
    auto core = torch::exp(-(psdf / lw).square());
    auto stroke = core * lg;

    auto dst = frame.index({S(y0, y1), S(x0, x1)});
    frame.index_put_({S(y0, y1), S(x0, x1)},
        dst - stroke.unsqueeze(2) * (255.0f - darkness));
}

void RobotArmRenderer::draw_local_sketch_box_rotated(torch::Tensor& frame, float cx, float cy,
                                                       float hx, float hy, float angle,
                                                       float lw, float darkness) {
    float extent = std::sqrt(hx * hx + hy * hy);
    float margin = lw * 4.0f + 3.0f;
    int y0, y1, x0, x1;
    if (!compute_bbox(cx, cy, extent, margin, W_, H_, y0, y1, x0, x1)) return;

    auto lx = x_.index({S(y0, y1), S(x0, x1)});
    auto ly = y_.index({S(y0, y1), S(x0, x1)});
    auto lw_ = wobble_.index({S(y0, y1), S(x0, x1)});
    auto lg = grain_.index({S(y0, y1), S(x0, x1)});

    auto sdf = sdf_box_rotated(lx, ly, cx, cy, hx, hy, angle);
    auto psdf = sdf + lw_;
    auto core = torch::exp(-(psdf / lw).square());
    auto stroke = core * lg;

    auto dst = frame.index({S(y0, y1), S(x0, x1)});
    frame.index_put_({S(y0, y1), S(x0, x1)},
        dst - stroke.unsqueeze(2) * (255.0f - darkness));
}

void RobotArmRenderer::draw_local_sketch_circle(torch::Tensor& frame, float cx, float cy,
                                                  float radius, float lw, float darkness) {
    float extent = radius;
    float margin = lw * 4.0f + 3.0f;
    int y0, y1, x0, x1;
    if (!compute_bbox(cx, cy, extent, margin, W_, H_, y0, y1, x0, x1)) return;

    auto lx = x_.index({S(y0, y1), S(x0, x1)});
    auto ly = y_.index({S(y0, y1), S(x0, x1)});
    auto lw_ = wobble_.index({S(y0, y1), S(x0, x1)});
    auto lg = grain_.index({S(y0, y1), S(x0, x1)});

    auto dist = ((lx - cx).square() + (ly - cy).square()).sqrt();
    auto sdf = dist - radius;
    auto psdf = sdf + lw_;
    auto core = torch::exp(-(psdf / lw).square());
    auto stroke = core * lg;

    auto dst = frame.index({S(y0, y1), S(x0, x1)});
    frame.index_put_({S(y0, y1), S(x0, x1)},
        dst - stroke.unsqueeze(2) * (255.0f - darkness));
}

void RobotArmRenderer::draw_local_color_pencil_box(torch::Tensor& frame, float cx, float cy,
                                                     float hx, float hy, float angle,
                                                     float col_r, float col_g, float col_b,
                                                     float lw, float outline_darkness) {
    float extent = std::sqrt(hx * hx + hy * hy);
    float margin = lw * 4.0f + 5.0f;  // extra margin for fill sigmoid edge
    int y0, y1, x0, x1;
    if (!compute_bbox(cx, cy, extent, margin, W_, H_, y0, y1, x0, x1)) return;

    auto lx = x_.index({S(y0, y1), S(x0, x1)});
    auto ly = y_.index({S(y0, y1), S(x0, x1)});
    auto lwo = wobble_.index({S(y0, y1), S(x0, x1)});
    auto lg = grain_.index({S(y0, y1), S(x0, x1)});

    auto sdf = sdf_box_rotated(lx, ly, cx, cy, hx, hy, angle);
    auto fill_mask = torch::sigmoid(-(sdf + lwo) * 3.0f);

    // Hatching
    float ca = std::cos(angle + 0.3f), sa = std::sin(angle + 0.3f);
    auto local_u = (lx - cx) * ca + (ly - cy) * sa;
    auto hatch1 = torch::sin(local_u * 1.8f) * 0.18f + 0.82f;
    auto local_v = -(lx - cx) * sa + (ly - cy) * ca;
    auto hatch2 = torch::sin(local_v * 2.5f) * 0.08f + 0.92f;

    auto alpha = (fill_mask * hatch1 * hatch2).unsqueeze(2);
    float mix = 0.55f;
    float fill_r = 255.0f * (1.0f - mix) + col_r * mix;
    float fill_g = 255.0f * (1.0f - mix) + col_g * mix;
    float fill_b = 255.0f * (1.0f - mix) + col_b * mix;

    auto color = torch::stack({
        torch::full_like(fill_mask, fill_b),
        torch::full_like(fill_mask, fill_g),
        torch::full_like(fill_mask, fill_r)
    }, 2);

    auto dst = frame.index({S(y0, y1), S(x0, x1)});
    dst = dst * (1.0f - alpha) + color * alpha;

    // Outline
    auto psdf = sdf + lwo;
    auto stroke = (torch::exp(-(psdf / lw).square()) * lg).unsqueeze(2);
    float dark_r = col_r * 0.35f, dark_g = col_g * 0.35f, dark_b = col_b * 0.35f;
    auto outline_color = torch::stack({
        torch::full_like(fill_mask, dark_b),
        torch::full_like(fill_mask, dark_g),
        torch::full_like(fill_mask, dark_r)
    }, 2);
    dst = dst * (1.0f - stroke) + outline_color * stroke;

    frame.index_put_({S(y0, y1), S(x0, x1)}, dst);
}

float RobotArmRenderer::rand_float(int seed) {
    uint32_t h = static_cast<uint32_t>(seed ^ rand_seed_);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return static_cast<float>(h & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

float RobotArmRenderer::smooth_step(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ── Construction ─────────────────────────────────────────────────────────

RobotArmRenderer::RobotArmRenderer(int width, int height, torch::Device device)
    : W_(width), H_(height), device_(device)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    y_ = torch::arange(0, H_, opts).unsqueeze(1).expand({H_, W_}).contiguous();
    x_ = torch::arange(0, W_, opts).unsqueeze(0).expand({H_, W_}).contiguous();

    // Pre-compute pencil textures (static — depend only on coordinates)
    wobble_ = pencil_wobble(x_, y_);
    grain_ = pencil_grain(x_, y_);

    // Table at 75% of screen height (compact, centered layout)
    table_y_ = H_ * 0.75f;

    // Robot base
    pedestal_h_ = H_ * 0.08f;
    pedestal_top_ = table_y_;
    base_x_ = W_ * 0.32f;
    base_y_ = table_y_ - pedestal_h_;

    // 3-link arm: link3 tilts slightly toward target.
    // IK solves for links 1+2 to reach the "wrist" (top of link3).
    link1_len_ = H_ * 0.28f;
    link2_len_ = H_ * 0.24f;
    link3_len_ = H_ * 0.14f;

    // Cube size
    cube_half_ = H_ * 0.042f;

    // Gripper geometry
    jaw_w_ = cube_half_ * 0.35f;
    jaw_h_ = cube_half_ * 1.0f;
    open_offset_ = cube_half_ + jaw_w_ * 0.5f + cube_half_ * 0.2f;
    closed_offset_ = cube_half_ + jaw_w_ * 0.5f;
    float rail_hh = jaw_w_ * 0.4f;
    float gap = 3.0f;
    // ee is at the bottom of link3. Grip offset from ee to cube center.
    grip_grab_offset_ = rail_hh + gap + cube_half_;

    // Stack position
    stack_x_ = W_ * 0.65f;

    // Home position: arm folded up well above table
    float home_tx = base_x_ + link1_len_ * 0.3f;
    float home_ty = base_y_ - link1_len_ * 0.7f;
    solve_ik(home_tx, home_ty, home_shoulder_, home_elbow_, home_wrist_);
    shoulder_angle_ = home_shoulder_;
    elbow_angle_ = home_elbow_;
    wrist_angle_ = home_wrist_;

    // Initialize cubes: [0]=R (left), [1]=S (center), [2]=O (right)
    // Order never changes; only spacing is randomized.
    for (int i = 0; i < 3; i++) {
        cubes_[i] = {0, 0, cube_half_, 0, 0, 0,
                     false, false, 0, 0, 0};
    }

    // Pre-render label bitmaps matching cube indices
    label_bmps_[0] = make_text_bitmap("R", 2, device_);
    label_bmps_[1] = make_text_bitmap("S", 2, device_);
    label_bmps_[2] = make_text_bitmap("O", 2, device_);

    // Cube colors (color pencil palette): R=warm red, S=teal, O=amber
    cubes_[0].r = 200; cubes_[0].g = 60;  cubes_[0].b = 60;   // R - red
    cubes_[1].r = 50;  cubes_[1].g = 140; cubes_[1].b = 140;  // S - teal
    cubes_[2].r = 210; cubes_[2].g = 160; cubes_[2].b = 50;   // O - amber

    randomize_cube_positions();

    state_ = State::Idle;
    state_timer_ = 0.5f;
}

// Compute 3 valid random x positions (sorted, with spacing) and store in scatter_target_x_.
void RobotArmRenderer::compute_scatter_targets() {
    rand_seed_ += 7;

    float max_reach = (link1_len_ + link2_len_) * 0.92f;
    float pedestal_w = H_ * 0.05f;
    float min_x = base_x_ + pedestal_w + cube_half_ * 2.5f;
    float max_x = base_x_ + max_reach;

    float table_left = W_ * 0.19f + cube_half_;
    float table_right = W_ * 0.81f - cube_half_;
    min_x = std::max(min_x, table_left);
    max_x = std::min(max_x, table_right);

    float min_gap = cube_half_ * 2.0f + jaw_w_ + cube_half_ * 0.5f;
    float range = max_x - min_x;

    for (int i = 0; i < 3; i++)
        scatter_target_x_[i] = min_x + rand_float(i * 31 + 5) * range;

    std::sort(scatter_target_x_.begin(), scatter_target_x_.end());

    for (int i = 1; i < 3; i++) {
        if (scatter_target_x_[i] - scatter_target_x_[i-1] < min_gap)
            scatter_target_x_[i] = scatter_target_x_[i-1] + min_gap;
    }
    if (scatter_target_x_[2] > max_x) {
        float shift = scatter_target_x_[2] - max_x;
        for (auto& tx : scatter_target_x_) tx -= shift;
        if (scatter_target_x_[0] < min_x) scatter_target_x_[0] = min_x;
        for (int i = 1; i < 3; i++) {
            if (scatter_target_x_[i] - scatter_target_x_[i-1] < min_gap)
                scatter_target_x_[i] = scatter_target_x_[i-1] + min_gap;
        }
    }
}

// Place cubes at scatter_target_x_ positions (order: left, center, right).
void RobotArmRenderer::randomize_cube_positions() {
    // On first call (constructor), compute targets ourselves
    if (scatter_target_x_[0] == 0 && scatter_target_x_[1] == 0 && scatter_target_x_[2] == 0)
        compute_scatter_targets();

    // Targets are sorted left-to-right, matching cube order [0]=R, [1]=S, [2]=O
    for (int i = 0; i < 3; i++) {
        cubes_[i].x = scatter_target_x_[i];
        cubes_[i].y = table_y_ - cube_half_;
        cubes_[i].stacked = false;
        cubes_[i].held = false;
        cubes_[i].vx = 0; cubes_[i].vy = 0;
        cubes_[i].angle = 0;
    }

    // Fixed pick order: [1]=S (center, base), [2]=O (right, 2nd), [0]=R (left, 3rd/top)
    // Stack bottom-to-top: S, O, R → reads top-to-bottom: "ROS"
    pick_order_ = {1, 2, 0};
    cubes_stacked_ = 0;
    current_pick_ = 0;
}

// ── 3-link IK ───────────────────────────────────────────────────────────
// Link3 tilts slightly toward the target for visible wrist articulation.
// The IK target (tx,ty) is the end-effector (bottom of link3).
// We compute a wrist world angle (slight tilt from vertical), then
// derive the wrist position and solve 2-link IK for links 1+2.

bool RobotArmRenderer::solve_ik(float tx, float ty, float& shoulder, float& elbow, float& wrist_world) {
    // Wrist world angle varies with both position and height for visible articulation.
    // Near base + high up: tilts toward horizontal. Far out + low: more vertical.
    float reach = link1_len_ + link2_len_;
    float horizontal_offset = (tx - base_x_) / reach;  // ~0.2 to ~0.9
    float height_factor = std::clamp((table_y_ - ty) / (table_y_ - base_y_), 0.0f, 2.0f);
    wrist_world = float(M_PI) * 0.5f + 0.5f * horizontal_offset - 0.5f * height_factor;

    // Wrist position (top of link3)
    float wx = tx - link3_len_ * std::cos(wrist_world);
    float wy = ty - link3_len_ * std::sin(wrist_world);

    float dx = wx - base_x_;
    float dy = wy - base_y_;
    float dist_sq = dx * dx + dy * dy;
    float dist = std::sqrt(dist_sq);

    float L1 = link1_len_, L2 = link2_len_;

    if (dist > L1 + L2 - 1.0f) {
        float scale = (L1 + L2 - 1.0f) / dist;
        dx *= scale; dy *= scale;
        dist_sq = dx * dx + dy * dy;
        dist = std::sqrt(dist_sq);
    }
    if (dist < std::abs(L1 - L2) + 1.0f) return false;

    float cos_elbow = (dist_sq - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
    cos_elbow = std::clamp(cos_elbow, -1.0f, 1.0f);
    elbow = std::acos(cos_elbow);

    float angle_to_target = std::atan2(dy, dx);
    float cos_alpha = (L1 * L1 + dist_sq - L2 * L2) / (2.0f * L1 * dist);
    cos_alpha = std::clamp(cos_alpha, -1.0f, 1.0f);
    shoulder = angle_to_target - std::acos(cos_alpha);

    return true;
}

// Forward kinematics: 3 links. Returns elbow, wrist, and ee positions.
void RobotArmRenderer::forward_kin(float shoulder, float elbow, float wrist_world,
                                    float& elbow_x, float& elbow_y,
                                    float& wrist_x, float& wrist_y,
                                    float& ee_x, float& ee_y) {
    elbow_x = base_x_ + link1_len_ * std::cos(shoulder);
    elbow_y = base_y_ + link1_len_ * std::sin(shoulder);
    float total_angle = shoulder + elbow;
    wrist_x = elbow_x + link2_len_ * std::cos(total_angle);
    wrist_y = elbow_y + link2_len_ * std::sin(total_angle);
    // Link3 uses wrist world angle (tilts slightly toward target)
    ee_x = wrist_x + link3_len_ * std::cos(wrist_world);
    ee_y = wrist_y + link3_len_ * std::sin(wrist_world);
}

// ── Motion helpers ──────────────────────────────────────────────────────

void RobotArmRenderer::start_move_to(float tx, float ty, float duration) {
    move_start_shoulder_ = shoulder_angle_;
    move_start_elbow_ = elbow_angle_;
    move_start_wrist_ = wrist_angle_;
    if (!solve_ik(tx, ty, move_target_shoulder_, move_target_elbow_, move_target_wrist_)) {
        move_target_shoulder_ = shoulder_angle_;
        move_target_elbow_ = elbow_angle_;
        move_target_wrist_ = wrist_angle_;
    }
    move_duration_ = duration;
    move_elapsed_ = 0.0f;
}

// ── Update (state machine) ──────────────────────────────────────────────

void RobotArmRenderer::update(float dt) {
    time_ += dt;

    if (gripper_opening_ < gripper_target_)
        gripper_opening_ = std::min(gripper_opening_ + GRIPPER_SPEED * dt, gripper_target_);
    else if (gripper_opening_ > gripper_target_)
        gripper_opening_ = std::max(gripper_opening_ - GRIPPER_SPEED * dt, gripper_target_);

    auto do_move = [&]() -> bool {
        move_elapsed_ += dt;
        float t = smooth_step(std::clamp(move_elapsed_ / move_duration_, 0.0f, 1.0f));
        shoulder_angle_ = move_start_shoulder_ + (move_target_shoulder_ - move_start_shoulder_) * t;
        elbow_angle_ = move_start_elbow_ + (move_target_elbow_ - move_start_elbow_) * t;
        wrist_angle_ = move_start_wrist_ + (move_target_wrist_ - move_start_wrist_) * t;
        return move_elapsed_ >= move_duration_;
    };

    float clearance = grip_grab_offset_ + cube_half_ * (2.5f + 2.0f * cubes_stacked_);
    float transit_ee_y = table_y_ - clearance;
    float grab_ee_y = table_y_ - cube_half_ - grip_grab_offset_;

    switch (state_) {
    case State::Idle:
        state_timer_ -= dt;
        if (state_timer_ <= 0.0f) {
            if (current_pick_ < 3) {
                if (cubes_stacked_ == 0) {
                    auto& c = cubes_[pick_order_[current_pick_]];
                    c.stacked = true;
                    stack_x_ = c.x;
                    cubes_stacked_++;
                    current_pick_++;
                    state_timer_ = 0.3f;
                } else {
                    auto& c = cubes_[pick_order_[current_pick_]];
                    start_move_to(c.x, transit_ee_y, 0.6f);
                    gripper_target_ = 1.0f;
                    state_ = State::MoveToAbove;
                }
            }
        }
        break;

    case State::MoveToAbove:
        if (do_move()) {
            auto& c = cubes_[pick_order_[current_pick_]];
            start_move_to(c.x, grab_ee_y, 0.4f);
            state_ = State::LowerToGrab;
        }
        break;

    case State::LowerToGrab:
        if (do_move()) {
            gripper_target_ = 0.0f;
            state_timer_ = 0.25f;
            state_ = State::CloseGripper;
        }
        break;

    case State::CloseGripper:
        state_timer_ -= dt;
        if (state_timer_ <= 0.0f) {
            cubes_[pick_order_[current_pick_]].held = true;
            auto& c = cubes_[pick_order_[current_pick_]];
            start_move_to(c.x, transit_ee_y, 0.35f);
            state_ = State::LiftCube;
        }
        break;

    case State::LiftCube:
        if (do_move()) {
            start_move_to(stack_x_, transit_ee_y, 0.6f);
            state_ = State::MoveToStack;
        }
        break;

    case State::MoveToStack:
        if (do_move()) {
            float stack_cube_y = table_y_ - cube_half_ - cubes_stacked_ * cube_half_ * 2.0f;
            float stack_ee_y = stack_cube_y - grip_grab_offset_;
            // Small random x wobble for imperfect stacking
            float wobble = (rand_float(current_pick_ * 13 + cubes_stacked_ * 7 + (int)(time_ * 10)) - 0.5f) * cube_half_ * 1.0f;
            start_move_to(stack_x_ + wobble, stack_ee_y, 0.35f);
            state_ = State::LowerToStack;
        }
        break;

    case State::LowerToStack:
        if (do_move()) {
            gripper_target_ = 1.0f;
            state_timer_ = 0.2f;
            state_ = State::OpenGripper;
        }
        break;

    case State::OpenGripper:
        state_timer_ -= dt;
        if (state_timer_ <= 0.0f) {
            auto& c = cubes_[pick_order_[current_pick_]];
            c.held = false;
            c.stacked = true;
            float stack_cube_y = table_y_ - cube_half_ - cubes_stacked_ * cube_half_ * 2.0f;
            // Use current ee_x (includes wobble from the move)
            float ex, ey, wx, wy, ee_x, ee_y;
            forward_kin(shoulder_angle_, elbow_angle_, wrist_angle_, ex, ey, wx, wy, ee_x, ee_y);
            c.x = ee_x;
            c.y = stack_cube_y;
            cubes_stacked_++;
            current_pick_++;
            start_move_to(stack_x_, transit_ee_y, 0.3f);
            state_ = State::LiftFromStack;
        }
        break;

    case State::LiftFromStack:
        if (do_move()) {
            if (current_pick_ >= 3) {
                move_start_shoulder_ = shoulder_angle_;
                move_start_elbow_ = elbow_angle_;
                move_start_wrist_ = wrist_angle_;
                move_target_shoulder_ = home_shoulder_;
                move_target_elbow_ = home_elbow_;
                move_target_wrist_ = home_wrist_;
                move_duration_ = 0.7f;
                move_elapsed_ = 0.0f;
                state_ = State::ReturnHome;
            } else {
                auto& c = cubes_[pick_order_[current_pick_]];
                start_move_to(c.x, transit_ee_y, 0.6f);
                state_ = State::MoveToAbove;
            }
        }
        break;

    case State::ReturnHome:
        if (do_move()) {
            state_timer_ = 0.8f;
            state_ = State::Collapse;
        }
        break;

    case State::Collapse:
        state_timer_ -= dt;
        if (state_timer_ <= 0.0f) {
            // Compute next cycle's sorted target positions
            compute_scatter_targets();

            // Targets are sorted left-to-right, matching cube order directly
            // (cube[0]=R always left, cube[1]=S always center, cube[2]=O always right)
            for (int i = 0; i < 3; i++) {
                cubes_[i].stacked = false;
                float dx = scatter_target_x_[i] - cubes_[i].x;
                cubes_[i].vy = -150.0f - rand_float(i * 23 + 1) * 60.0f;
                // Scale horizontal speed by flight time so steeper launches
                // travel less horizontally, arriving near the target in one arc.
                float flight_time = 2.0f * std::abs(cubes_[i].vy) / 500.0f;
                cubes_[i].vx = dx / std::max(flight_time * 1.8f, 0.5f);
                cubes_[i].angle = 0;
            }
            state_timer_ = 2.0f;
            state_ = State::Scatter;
        }
        break;

    case State::Scatter: {
        float gravity = 500.0f;
        for (int i = 0; i < 3; i++) {
            auto& c = cubes_[i];
            c.vy += gravity * dt;
            c.x += c.vx * dt;
            c.y += c.vy * dt;

            // Steer gently toward target x (soft spring + critical damping)
            float pull = (scatter_target_x_[i] - c.x) * 2.0f;
            c.vx += pull * dt;
            c.vx *= std::exp(-1.5f * dt);  // moderate damping to prevent oscillation

            if (c.y > table_y_ - cube_half_) {
                c.y = table_y_ - cube_half_;
                c.vy = -c.vy * 0.25f;
                c.vx *= 0.6f;
                if (std::abs(c.vy) < 8.0f) c.vy = 0;
            }
            float max_reach = (link1_len_ + link2_len_) * 0.92f;
            float pedestal_w = H_ * 0.05f;
            float left_edge = std::max(base_x_ + pedestal_w + cube_half_ * 2.5f,
                                       W_ * 0.19f + cube_half_);
            float right_edge = std::min(base_x_ + max_reach,
                                        W_ * 0.81f - cube_half_);
            if (c.x < left_edge) { c.x = left_edge; c.vx = std::abs(c.vx) * 0.3f; }
            if (c.x > right_edge) { c.x = right_edge; c.vx = -std::abs(c.vx) * 0.3f; }
        }
        state_timer_ -= dt;
        if (state_timer_ <= 0.0f) {
            // Settle cubes where they are — labels and order unchanged
            for (auto& c : cubes_) {
                c.y = table_y_ - cube_half_;
                c.vx = 0; c.vy = 0;
                c.angle = 0;
                c.stacked = false;
                c.held = false;
            }
            // Fixed pick order: [1]=S (center, base), [2]=O (right, 2nd), [0]=R (left, 3rd/top)
            pick_order_ = {1, 2, 0};
            cubes_stacked_ = 0;
            current_pick_ = 0;
            state_timer_ = 0.5f;
            state_ = State::Idle;
        }
        break;
    }
    }

    // Track held cube to gripper
    for (auto& c : cubes_) {
        if (c.held) {
            float ex, ey, wx, wy, ee_x, ee_y;
            forward_kin(shoulder_angle_, elbow_angle_, wrist_angle_, ex, ey, wx, wy, ee_x, ee_y);
            c.x = ee_x;
            c.y = ee_y + grip_grab_offset_;
        }
    }
}

// ── Background: pure white ───────────────────────────────────────────────

torch::Tensor RobotArmRenderer::render_background() {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    return torch::full({H_, W_, 3}, 255.0f, opts);
}

// ── Table ───────────────────────────────────────────────────────────────

void RobotArmRenderer::render_table(torch::Tensor& frame) {
    float table_top = table_y_;
    float table_left = W_ * 0.15f;
    float table_right = W_ * 0.85f;
    float leg_bottom = H_ * 0.96f;

    float thin_lw = 1.0f;
    int n_strokes = 3;
    float margin = thin_lw * 4.0f + 3.0f;  // wobble ~1px + stroke falloff

    // Table surface — localized to a thin horizontal band
    int surf_y0 = std::max(0, (int)std::floor(table_top - margin));
    int surf_y1 = std::min(H_, (int)std::ceil(table_top + margin));
    int surf_x0 = std::max(0, (int)std::floor(table_left - margin));
    int surf_x1 = std::min(W_, (int)std::ceil(table_right + margin));
    if (surf_y0 < surf_y1 && surf_x0 < surf_x1) {
        auto lx = x_.index({S(surf_y0, surf_y1), S(surf_x0, surf_x1)});
        auto ly = y_.index({S(surf_y0, surf_y1), S(surf_x0, surf_x1)});
        auto lg = grain_.index({S(surf_y0, surf_y1), S(surf_x0, surf_x1)});
        auto wobble_h = torch::sin(lx * 0.05f) * 0.6f + torch::sin(lx * 0.03f + 0.7f) * 0.4f;
        auto x_mask = torch::sigmoid((lx - table_left) * 2.0f) * torch::sigmoid((table_right - lx) * 2.0f);

        auto dst = frame.index({S(surf_y0, surf_y1), S(surf_x0, surf_x1)});
        for (int s = 0; s < n_strokes; s++) {
            float offset = (s - (n_strokes - 1) * 0.5f) * 0.8f;
            auto sdf = (ly - (table_top + offset) - wobble_h).abs();
            auto surface_stroke = pencil_stroke_raw(sdf, thin_lw, lg) * x_mask;
            dst = dst - surface_stroke.unsqueeze(2) * (255.0f - PENCIL_LIGHT);
        }
        frame.index_put_({S(surf_y0, surf_y1), S(surf_x0, surf_x1)}, dst);
    }

    // Two legs — localized to thin vertical bands
    for (float leg_x : {table_left + W_ * 0.05f, table_right - W_ * 0.05f}) {
        int lx0 = std::max(0, (int)std::floor(leg_x - margin));
        int lx1 = std::min(W_, (int)std::ceil(leg_x + margin));
        int ly0 = std::max(0, (int)std::floor(table_top - margin));
        int ly1 = std::min(H_, (int)std::ceil(leg_bottom + margin));
        if (lx0 >= lx1 || ly0 >= ly1) continue;

        auto rlx = x_.index({S(ly0, ly1), S(lx0, lx1)});
        auto rly = y_.index({S(ly0, ly1), S(lx0, lx1)});
        auto rlg = grain_.index({S(ly0, ly1), S(lx0, lx1)});
        auto wobble_v = torch::sin(rly * 0.07f) * 0.6f + torch::sin(rly * 0.04f + 0.7f) * 0.4f;
        auto y_mask = torch::sigmoid((rly - table_top) * 2.0f) * torch::sigmoid((leg_bottom - rly) * 2.0f);

        auto dst = frame.index({S(ly0, ly1), S(lx0, lx1)});
        for (int s = 0; s < n_strokes; s++) {
            float offset = (s - (n_strokes - 1) * 0.5f) * 0.8f;
            auto sdf = (rlx - (leg_x + offset) - wobble_v).abs();
            auto leg_stroke = pencil_stroke_raw(sdf, thin_lw, rlg) * y_mask;
            dst = dst - leg_stroke.unsqueeze(2) * (255.0f - PENCIL_LIGHT);
        }
        frame.index_put_({S(ly0, ly1), S(lx0, lx1)}, dst);
    }
}

// ── Render a single cube with label ─────────────────────────────────────

void RobotArmRenderer::render_cubes(torch::Tensor& frame) {
    for (int i = 0; i < 3; i++) {
        auto& c = cubes_[i];
        if (c.held) continue;

        // Color pencil fill with hatching + colored outline (localized)
        draw_local_color_pencil_box(frame, c.x, c.y, c.size, c.size, c.angle,
                                     c.r, c.g, c.b, 1.8f, PENCIL_DARK);

        // Dark label (white on colored background)
        auto& bmp = label_bmps_[i];
        int bh = bmp.size(0), bw = bmp.size(1);
        int lx = static_cast<int>(c.x) - bw / 2;
        int ly = static_cast<int>(c.y) - bh / 2;
        int y0 = std::max(0, ly), y1 = std::min(H_, ly + bh);
        int x0 = std::max(0, lx), x1 = std::min(W_, lx + bw);
        if (y0 < y1 && x0 < x1) {
            auto region = bmp.index({S(y0 - ly, y1 - ly), S(x0 - lx, x1 - lx)});
            auto alpha = region.unsqueeze(2);
            auto dst = frame.index({S(y0, y1), S(x0, x1)});
            frame.index_put_({S(y0, y1), S(x0, x1)},
                dst * (1.0f - alpha) + alpha * 255.0f);
        }
    }
}

// ── Robot arm (3 links) ─────────────────────────────────────────────────

void RobotArmRenderer::render_arm(torch::Tensor& frame) {
    float lw = 1.8f;

    float elbow_x, elbow_y, wrist_x, wrist_y, ee_x, ee_y;
    forward_kin(shoulder_angle_, elbow_angle_, wrist_angle_, elbow_x, elbow_y, wrist_x, wrist_y, ee_x, ee_y);

    // ── Base pedestal ──
    float pedestal_w = H_ * 0.05f;
    float ped_cy = pedestal_top_ - pedestal_h_ * 0.5f;
    draw_local_sketch_box(frame, base_x_, ped_cy, pedestal_w, pedestal_h_ * 0.5f,
                           lw, PENCIL_DARK);

    // ── Link 1 (upper arm) ──
    float l1_cx = (base_x_ + elbow_x) * 0.5f;
    float l1_cy = (base_y_ + elbow_y) * 0.5f;
    float l1_angle = std::atan2(elbow_y - base_y_, elbow_x - base_x_);
    float arm_w1 = H_ * 0.018f;
    draw_local_sketch_box_rotated(frame, l1_cx, l1_cy, link1_len_ * 0.5f, arm_w1, l1_angle,
                                   lw, PENCIL_DARK);

    // ── Link 2 (forearm) ──
    float l2_cx = (elbow_x + wrist_x) * 0.5f;
    float l2_cy = (elbow_y + wrist_y) * 0.5f;
    float l2_angle = std::atan2(wrist_y - elbow_y, wrist_x - elbow_x);
    float arm_w2 = H_ * 0.015f;
    draw_local_sketch_box_rotated(frame, l2_cx, l2_cy, link2_len_ * 0.5f, arm_w2, l2_angle,
                                   lw, PENCIL_MED);

    // ── Link 3 (wrist-to-ee, tilts with wrist angle) ──
    float l3_cx = (wrist_x + ee_x) * 0.5f;
    float l3_cy = (wrist_y + ee_y) * 0.5f;
    float l3_angle = std::atan2(ee_y - wrist_y, ee_x - wrist_x);
    float arm_w3 = H_ * 0.012f;
    draw_local_sketch_box_rotated(frame, l3_cx, l3_cy, link3_len_ * 0.5f, arm_w3, l3_angle,
                                   1.4f, PENCIL_MED);

    // ── Joint circles ──
    float j1_r = H_ * 0.015f;
    draw_local_sketch_circle(frame, base_x_, base_y_, j1_r, lw, PENCIL_DARK);

    float j2_r = H_ * 0.012f;
    draw_local_sketch_circle(frame, elbow_x, elbow_y, j2_r, lw, PENCIL_DARK);

    float j3_r = H_ * 0.010f;
    draw_local_sketch_circle(frame, wrist_x, wrist_y, j3_r, 1.4f, PENCIL_MED);

    float j4_r = H_ * 0.009f;
    draw_local_sketch_circle(frame, ee_x, ee_y, j4_r, 1.4f, PENCIL_MED);

    // ── Gripper ──
    render_gripper(frame, ee_x, ee_y);
}

// ── Gripper ─────────────────────────────────────────────────────────────

void RobotArmRenderer::render_gripper(torch::Tensor& frame, float ex, float ey) {
    float lw = 1.4f;
    float offset = closed_offset_ + gripper_opening_ * (open_offset_ - closed_offset_);

    // Horizontal rail
    float rail_hw = open_offset_ + jaw_w_ * 0.5f;
    float rail_hh = jaw_w_ * 0.4f;
    draw_local_sketch_box(frame, ex, ey, rail_hw, rail_hh, lw, PENCIL_MED);

    // Two jaws
    float jaw_top = ey + rail_hh;
    float jaw_cy = jaw_top + jaw_h_ * 0.5f;

    for (float side : {-1.0f, 1.0f}) {
        float jaw_cx = ex + offset * side;
        draw_local_sketch_box(frame, jaw_cx, jaw_cy, jaw_w_ * 0.5f, jaw_h_ * 0.5f,
                               lw, PENCIL_MED);
    }
}

// ── Frame composition ───────────────────────────────────────────────────

torch::Tensor RobotArmRenderer::render_frame() {
    torch::NoGradGuard no_grad;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    auto frame = render_background();
    render_table(frame);
    render_cubes(frame);  // non-held cubes
    render_arm(frame);

    // Held cubes on top of arm
    for (int i = 0; i < 3; i++) {
        auto& c = cubes_[i];
        if (!c.held) continue;

        draw_local_color_pencil_box(frame, c.x, c.y, c.size, c.size, c.angle,
                                     c.r, c.g, c.b, 1.8f, PENCIL_DARK);

        // White label on colored background
        auto& bmp = label_bmps_[i];
        int bh = bmp.size(0), bw = bmp.size(1);
        int lx = static_cast<int>(c.x) - bw / 2;
        int ly = static_cast<int>(c.y) - bh / 2;
        int y0 = std::max(0, ly), y1 = std::min(H_, ly + bh);
        int x0 = std::max(0, lx), x1 = std::min(W_, lx + bw);
        if (y0 < y1 && x0 < x1) {
            auto region = bmp.index({S(y0 - ly, y1 - ly), S(x0 - lx, x1 - lx)});
            auto alpha = region.unsqueeze(2);
            auto dst = frame.index({S(y0, y1), S(x0, x1)});
            frame.index_put_({S(y0, y1), S(x0, x1)},
                dst * (1.0f - alpha) + alpha * 255.0f);
        }
    }

    frame = torch::clamp(frame, 0.0f, 255.0f);
    auto alpha_channel = torch::full({H_, W_, 1}, 255.0f, opts);
    frame = torch::cat({frame, alpha_channel}, 2);
    frame = frame.to(torch::kUInt8).contiguous();
    return frame;
}
