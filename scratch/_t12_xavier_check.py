# -*- coding: utf-8 -*-
"""[T12 探针3] 复现 main.make_model 的 xavier 循环：验证排除 prefix_embed 后先验是否保留。只读。"""
import io
import os
import sys

sys.path.insert(0, os.path.abspath("."))
_buf = io.StringIO()
_old = sys.stdout
sys.stdout = _buf
from src.utils.config import config  # noqa: E402
sys.stdout = _old

config.dataset = "ED"
config.hidden_dim = 300
config.emb_dim = 300
config.hop = 1
config.heads = 2
config.depth = 100
config.filter = 300
config.dropout = 0.1
config.PAD_idx = 1
config.lr = 0.0001
config.noam = 0
config.weight_sharing = False
config.use_pcam = True
config.use_emo_bias = True
config.use_mcp = True
config.device = "cpu"
config.test = False
config.pretrain_emb = True
config.save_path = "save/_t12_probe"

import torch  # noqa: E402
from torch.nn.init import xavier_uniform_  # noqa: E402


class MockVocab:
    def __init__(self):
        self.word2idx = {"<pad>": 1, "<unk>": 0, "<s>": 2, "</s>": 3}
        for i in range(1000):
            self.word2idx[f"word_{i}"] = i + 4
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word2index = self.word2idx
        self.n_words = len(self.word2idx)

    def __len__(self):
        return len(self.word2idx)


vocab = MockVocab()
from src.models.CASE.model import CASE  # noqa: E402

config.use_proto_prefix = True
config.proto_prefix_len = 4
config.proto_prefix_init = "proto"
config.proto_prefix_init_npz = ""
config.proto_prefix_source = "ckpt"
config.proto_prefix_prior_path = "results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best"

torch.manual_seed(13)
m = CASE(vocab, emotion_num=32, strategy_num=8)
pe0 = m.decoder.dec[0].prefix_embed.detach().clone()

# 旧逻辑（bug）：不排除 prefix_embed
torch.manual_seed(13)
m_bug = CASE(vocab, emotion_num=32, strategy_num=8)
for n, p in m_bug.named_parameters():
    if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
        xavier_uniform_(p)
pe_bug = m_bug.decoder.dec[0].prefix_embed.detach().clone()

# 新逻辑（修复）：排除 prefix_embed
torch.manual_seed(13)
from main import make_model
m_fix = make_model(vocab, 32, 8)
pe_fix = m_fix.decoder.dec[0].prefix_embed.detach().clone()

print(f"init  vs old-xavier : max|d|={(pe0 - pe_bug).abs().max().item():.6e}   <- 旧逻辑把先验冲掉了")
print(f"init  vs new-xavier : max|d|={(pe0 - pe_fix).abs().max().item():.6e}   <- 修复后保留")
print(f"old-xavier 是否等范数 xavier 初值: L2={[round(v,3) for v in pe_bug.norm(dim=1).tolist()]}")
print(f"init L2={[round(v,3) for v in pe0.norm(dim=1).tolist()]}")

assert torch.equal(pe0, pe_fix), "P9: actual make_model overwrote the prior"
