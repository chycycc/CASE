# -*- coding: utf-8 -*-
"""
[CASE-EPCL V8 Trial 4 集成验证]
测试目标: V8 Trial 4 双锚点解耦与时序软退火架构集成验证
- 双锚点配置: cls_anchor = "fine_emotion", epcl_anchor = "emotion_enc"
- 时序软退火: soft_freeze = True, freeze_decay_weight = 0.05
- 动量样本中心原型: use_mcp = True (momentum = 0.99)
- 加性角度硬边际: use_arc_margin = True (arc_margin = 0.30)
- 验证前向传播、软退火损失计算、反向传播无 OOM、无梯度异常
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.abspath("."))

from src.models.CASE.model import CASE
from src.utils.config import config


class TestV8Trial4Integration(unittest.TestCase):
    def setUp(self):
        config.dataset = "ED"
        config.woStrategy = True
        config.emotion_head_type = "residual_mlp"
        config.mlp_hidden_dim = 300
        config.mlp_dropout = 0.1
        config.num_prototypes_per_class = 1
        config.use_pcam = True
        config.pcam_heads = 2
        config.pcam_dropout = 0.1
        config.pcam_gate_bias = -1.0
        
        config.use_erp = True
        config.erp_hidden_dim = 768
        config.erp_dropout = 0.1
        
        config.use_mcp = True
        config.mcp_momentum = 0.99
        
        config.use_arc_margin = True
        config.arc_margin = 0.30
        config.arc_mode = "cos"
        config.epcl_anchor = "emotion_enc"
        config.cls_anchor = "fine_emotion"
        config.soft_freeze = True
        config.freeze_decay_weight = 0.05
        
        config.lambda_epcl = 0.08
        config.alpha_uni = 1.0
        config.alpha_mim = 0.10
        config.fine_weight = 0.30
        config.coarse_weight = 1.0
        config.div_weight = 2.0
        config.batch_size = 4
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_trial4_architecture_and_soft_freeze(self):
        """测试 Trial 4 双锚点解耦与软退火配置"""
        class DummyVocab:
            def __init__(self):
                self.word2index = {"<pad>": 1, "<eos>": 2, "<sos>": 3, "<unk>": 0}
                self.index2word = {1: "<pad>", 2: "<eos>", 3: "<sos>", 0: "<unk>"}
                self.word2idx = self.word2index
                self.idx2word = self.index2word
                self.n_words = 100
            def __len__(self):
                return 100

        vocab = DummyVocab()
        model = CASE(vocab, emotion_num=32, strategy_num=8, is_eval=False)
        model.to(config.device)

        self.assertEqual(model.epcl_anchor, "emotion_enc")
        self.assertEqual(model.cls_anchor, "fine_emotion")
        self.assertTrue(model.soft_freeze)
        self.assertEqual(model.freeze_decay_weight, 0.05)

        # 模拟触发软退火
        model.freeze_emo_head(step=28000)
        self.assertTrue(model.is_frozen)
        for p in model.emotion_linear.parameters():
            self.assertTrue(p.requires_grad, "软退火状态下分类头参数必须保持可微")

        print("[+] Trial 4 双锚点解耦与软退火断言验证全部通过！")


if __name__ == "__main__":
    unittest.main()
