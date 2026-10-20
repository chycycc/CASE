# -*- coding: utf-8 -*-
"""[T12 独立探针2] 用 mock vocab 构造 CASE(proto) 与 CASE(random)，比较 prefix_embed 实际数值。只读。"""
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
config.save_path = "save/_t12_probe"

import torch  # noqa: E402


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


def build(mode, npz=""):
    config.use_proto_prefix = True
    config.proto_prefix_len = 4
    config.proto_prefix_init = mode
    config.proto_prefix_init_npz = npz
    config.proto_prefix_source = "ckpt"
    config.proto_prefix_prior_path = "results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best"
    torch.manual_seed(13)
    m = CASE(vocab, emotion_num=32, strategy_num=8)
    pe = m.decoder.dec[0].prefix_embed.detach().clone()
    return m, pe


torch.manual_seed(13)
m_on, pe_on = build("proto")
torch.manual_seed(13)
m_rand, pe_rand = build("random", "save/test_trial12/proto_prefix_init.npz")

print("prefix_embed shapes:", tuple(pe_on.shape), tuple(pe_rand.shape))
print("on  [0,:5] :", [round(v, 5) for v in pe_on[0, :5].tolist()])
print("rand[0,:5] :", [round(v, 5) for v in pe_rand[0, :5].tolist()])
print(f"max|E_on - E_rand| = {(pe_on - pe_rand).abs().max().item():.6e}")
print(f"on  row L2 = {[round(v,4) for v in pe_on.norm(dim=1).tolist()]}")
print(f"rand row L2 = {[round(v,4) for v in pe_rand.norm(dim=1).tolist()]}")
print("use_proto_prefix (CASE/DecoderLayer):",
      m_on.use_proto_prefix, m_on.decoder.dec[0].use_proto_prefix,
      m_rand.use_proto_prefix, m_rand.decoder.dec[0].use_proto_prefix)
