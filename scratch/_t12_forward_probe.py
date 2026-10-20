# -*- coding: utf-8 -*-
"""[T12 独立探针] 前缀是否真正影响 self-attn 与 DecoderLayer 输出。只读，不改项目文件。"""
import io
import os
import sys

sys.path.insert(0, os.path.abspath("."))
_buf = io.StringIO()
_old = sys.stdout
sys.stdout = _buf
from src.utils.config import config  # noqa: E402
sys.stdout = _old

config.hidden_dim = 300
config.heads = 2
config.depth = 100
config.filter = 300
config.dropout = 0.1
config.PAD_idx = 1
config.device = "cpu"

import torch  # noqa: E402
from src.models.common import DecoderLayer, MultiHeadAttention  # noqa: E402

torch.manual_seed(0)
B, L = 2, 6
q = torch.randn(B, L, 300)
mask = torch.zeros(B, L, L, dtype=torch.bool)

mha = MultiHeadAttention(300, 100, 100, 300, 2, None, 0.0)
o0, _ = mha(q, q, q, mask)
e1 = torch.randn(4, 300)
e2 = torch.randn(4, 300)
o1, _ = mha(q, q, q, mask, extra_kv=e1)
o2, _ = mha(q, q, q, mask, extra_kv=e2)
print(f"[MHA] no-prefix vs prefix : max|d|={(o0 - o1).abs().max().item():.3e}")
print(f"[MHA] prefix1  vs prefix2 : max|d|={(o1 - o2).abs().max().item():.3e}")

enc = torch.randn(B, L, 300)
enc_mask = torch.zeros(B, 1, L, dtype=torch.bool)
inputs = (torch.randn(B, L, 300), enc, None,
          (enc_mask, mask),
          torch.randn(B, L, 300), enc_mask,
          torch.randn(B, L, 300), enc_mask)


def run(init):
    dl = DecoderLayer(300, 100, 100, 300, 2, None, 0.0, 0.0, 0.0,
                      use_proto_prefix=True, proto_prefix_len=4, proto_init=init)
    y, *_ = dl(inputs)
    return y


dl_on = run(e1)
dl_rand = run(e2)
print(f"[DecoderLayer] prefix1 vs prefix2 : max|d|={(dl_on - dl_rand).abs().max().item():.3e}")
print(f"[DecoderLayer] use_proto_prefix attr = {DecoderLayer.__init__ is not None}")
