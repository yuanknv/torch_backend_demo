#pragma once
#include <torch/torch.h>
#include <string>

// Render a text string into an anti-aliased [H, W] float bitmap (0..1).
// scale: screen pixels per glyph pixel.
torch::Tensor make_text_bitmap(const std::string& text, int scale, torch::Device dev);
