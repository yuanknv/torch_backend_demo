#include "font.h"
#include <array>
#include <unordered_map>
#include <cmath>

// ── Bitmap font (10x14, rounded) ────────────────────────────────────────
// Each glyph: 14 rows, 10 bits per row (stored as uint16_t, bit 9 = leftmost).
static constexpr int GLYPH_W = 10, GLYPH_H = 14;

static const std::array<uint16_t, 14>& glyph_for(char c) {
    static const std::unordered_map<char, std::array<uint16_t, 14>> font = {
        // Hand-drawn style R — slightly wobbly strokes
        {'R', {0b1111110000,
               0b1101111100,
               0b1100001100,
               0b1100001100,
               0b1100011100,
               0b1111111000,
               0b1111100000,
               0b1101110000,
               0b1100111000,
               0b1100011100,
               0b1100001100,
               0b1100001110,
               0b1100000110,
               0b0100000110}},
        // Hand-drawn style O — slightly asymmetric oval
        {'O', {0b0001110000,
               0b0111111100,
               0b0110001100,
               0b1100000110,
               0b1100000110,
               0b1100000110,
               0b1100000110,
               0b1100000110,
               0b1100000110,
               0b1100000110,
               0b0110001100,
               0b0110001100,
               0b0011111000,
               0b0001110000}},
        // Hand-drawn style S — fluid curves
        {'S', {0b0011111000,
               0b0111111100,
               0b1110000100,
               0b1100000000,
               0b0111000000,
               0b0011110000,
               0b0001111100,
               0b0000011110,
               0b0000000110,
               0b0000000110,
               0b0100001110,
               0b1110011100,
               0b0111111000,
               0b0011110000}},
    };
    static const std::array<uint16_t, 14> blank = {};
    auto it = font.find(c);
    return it != font.end() ? it->second : blank;
}

torch::Tensor make_text_bitmap(const std::string& text, int scale, torch::Device dev) {
    int gw = GLYPH_W, gh = GLYPH_H, gap = 2;
    // Supersample at 4× the glyph resolution for smooth results
    int ss = 4;
    int raw_w = (int)text.size() * (gw + gap) - gap;
    int raw_h = gh;
    int tw = raw_w * scale;
    int th = raw_h * scale;

    // Build at 1:1 glyph resolution
    auto glyph_bmp = torch::zeros({raw_h, raw_w}, torch::kFloat32);
    auto acc = glyph_bmp.accessor<float, 2>();
    for (int ci = 0; ci < (int)text.size(); ci++) {
        auto& g = glyph_for(text[ci]);
        int ox = ci * (gw + gap);
        for (int gy = 0; gy < gh; gy++)
            for (int gx = 0; gx < gw; gx++)
                if (g[gy] & (1 << (9 - gx)))
                    acc[gy][ox + gx] = 1.0f;
    }

    // First upscale to supersample resolution (4× larger than final)
    auto input = glyph_bmp.unsqueeze(0).unsqueeze(0);
    int ss_h = th * ss, ss_w = tw * ss;
    auto supersampled = torch::nn::functional::interpolate(input,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ss_h, ss_w})
            .mode(torch::kBilinear)
            .align_corners(false));

    // Threshold at supersample resolution for clean edges
    auto ss_bmp = supersampled.squeeze(0).squeeze(0);
    ss_bmp = (ss_bmp > 0.35f).to(torch::kFloat32);

    // Downsample to final resolution — averaging gives smooth antialiased edges
    auto ss_input = ss_bmp.unsqueeze(0).unsqueeze(0);
    auto final_bmp = torch::nn::functional::interpolate(ss_input,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{th, tw})
            .mode(torch::kBilinear)
            .align_corners(false));
    auto bmp = final_bmp.squeeze(0).squeeze(0);

    // Gentle contrast boost while preserving smooth edges
    float floor_val = 1.0f / (1.0f + std::exp(0.25f * 6.0f));
    bmp = (torch::sigmoid((bmp - 0.25f) * 6.0f) - floor_val) / (1.0f - floor_val);
    bmp = torch::clamp(bmp, 0.0f, 1.0f);
    return bmp.to(dev);
}
