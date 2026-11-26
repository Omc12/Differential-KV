"""The residual-attention weighting must rotate with the TEXT decoder's RoPE.

Ministral-3 (a vision-language model) lists its vision tower's 2-D rotary
first; taking "the first *RotaryEmbedding module" rotated text keys with patch
tables and crashed at 32k with a device-side gather out of bounds.
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)


class PixtralRotaryEmbedding(nn.Module):           # the vision tower's: tiny table
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.zeros(16, 8))

    def forward(self, x, position_ids):
        return self.table[position_ids], self.table[position_ids]     # OOB past 16


class TextRotaryEmbedding(nn.Module):
    def forward(self, x, position_ids):
        c = torch.ones(position_ids.shape + (8,))
        return c, torch.zeros_like(c)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.rotary_emb = TextRotaryEmbedding()


class _VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.rotary = PixtralRotaryEmbedding()     # first in named_modules
        self.language_model = _Decoder()

    def get_decoder(self):
        return self.language_model


def test_residual_weighting_uses_the_text_rotary():
    from runtime.dkv_attention import _find_rotary
    rot = _find_rotary(_VLM())
    assert rot is not None
    cos, _ = rot(torch.zeros(1), torch.arange(32768).unsqueeze(0))   # no OOB
    assert cos.shape[-2] == 32768
