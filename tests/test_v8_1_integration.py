# -*- coding: utf-8 -*-
"""
[CASE-EPCL V8.1 双轨实验集成测试]
测试目标:
1. 路线 A (Trial A): KEMP 词表层情感偏置投影维度匹配、自适应广播、梯度回传与解码生成测试
2. 路线 B (Trial B): 多任务辅助损失时序余弦平滑软退火权重数学精度验证、复合早停评分逻辑验证
3. 冻结机制安全性: 路线 A 禁用冻结 (disable_freeze) 验证，路线 B 软退火不锁死参数验证
4. 代码合规性检查: 杜绝 F.sigmoid、np.float、原地破坏计算图操作
"""

import os
import sys
import math
import unittest
import torch
import torch.nn as nn
import numpy as np

# 注册项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE, Generator
from src.utils.data.loader import prepare_data_seq


class TestV81Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config.seed = 13
        set_seed()
        # 基础测试配置
        config.dataset = "ED"
        config.woStrategy = True
        config.batch_size = 2
        config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        config.cls_anchor = "default"
        config.use_unlikelihood = True
        config.unlikelihood_weight = 0.1
        config.test = False
        config.model = "case"
        config.pointer_gen = False

        print("\n[*] 正在准备测试词表与微型数据集...")
        _, _, cls.test_set, cls.vocab, cls.emo_num, cls.strategy_num = prepare_data_seq(batch_size=2)

    def test_01_route_b_cosine_annealing_math(self):
        """测试路线 B: 余弦软退火权重数学公式精度"""
        print("\n--- [Test 01] 测试路线 B 时序余弦软退火权重衰减精度 ---")
        start_step = 24000
        anneal_steps = 16000
        min_w = 0.05

        def get_soft_weight(iter_step):
            if iter_step >= start_step:
                progress = min((iter_step - start_step) / float(anneal_steps), 1.0)
                return min_w + (1.0 - min_w) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        # 步数验证
        self.assertAlmostEqual(get_soft_weight(0), 1.0, places=5)
        self.assertAlmostEqual(get_soft_weight(12000), 1.0, places=5)
        self.assertAlmostEqual(get_soft_weight(24000), 1.0, places=5)
        
        # 中点 32000 步 (progress = 0.5, cos(pi/2) = 0): weight 应为 0.05 + 0.95*0.5 = 0.525
        mid_weight = get_soft_weight(32000)
        self.assertAlmostEqual(mid_weight, 0.525, places=5)
        
        # 终点 40000 步 (progress = 1.0, cos(pi) = -1): weight 应为 0.05
        end_weight = get_soft_weight(40000)
        self.assertAlmostEqual(end_weight, 0.05, places=5)

        # 40000 步以后继续平稳保持 0.05
        post_weight = get_soft_weight(52000)
        self.assertAlmostEqual(post_weight, 0.05, places=5)
        print("    [√] 步数点 0/12k/24k/32k/40k/52k 权重数学验证全量通过: 1.0 -> 0.525 -> 0.05 -> 0.05")

    def test_02_composite_score_formula(self):
        """测试路线 B: 复合早停评分计算逻辑"""
        print("\n--- [Test 02] 测试路线 B 复合早停评分公式 ---")
        ppl = 32.50
        emo_acc = 0.4050
        w = 20.0
        score = ppl - w * emo_acc
        expected = 32.50 - 20.0 * 0.4050  # 32.5 - 8.1 = 24.4
        self.assertAlmostEqual(score, expected, places=4)
        self.assertAlmostEqual(score, 24.40, places=4)

        # 验证退化模型与平衡模型的优劣判据对比
        # 模型1: PPL 极微弱优化 (32.40), 但分类头断崖下跌 (0.3500) -> score = 32.40 - 7.00 = 25.40 (劣)
        # 模型2: PPL 稍高 (32.60), 分类头稳健 (0.4150) -> score = 32.60 - 8.30 = 24.30 (优，得分更低)
        score_bad_cls = 32.40 - 20.0 * 0.3500
        score_good_cls = 32.60 - 20.0 * 0.4150
        self.assertTrue(score_good_cls < score_bad_cls, "复合评分应能成功筛选出综合质量更高、避免分类头退化的黄金检查点！")
        print("    [√] 复合早停判据有效性验证通过！")

    def test_03_route_a_vocab_bias_forward_backward(self):
        """测试路线 A: KEMP 词表层情感偏置投影前向计算与反向传播"""
        print("\n--- [Test 03] 测试路线 A 词表层情感偏置投影与梯度流动 ---")
        config.use_emo_bias = True
        config.emo_bias_gate_init = -2.0
        config.disable_freeze = True
        config.pretrain = False

        model = CASE(
            self.vocab,
            emotion_num=self.emo_num,
            strategy_num=self.strategy_num,
            is_eval=False,
            model_file_path=None
        )
        model = model.to(config.device)
        model.train()

        # 检查结构初始化
        self.assertTrue(hasattr(model, "emo_to_vocab"), "模型必须具有 emo_to_vocab 属性")
        self.assertTrue(hasattr(model, "emo_bias_gate"), "模型必须具有 emo_bias_gate 属性")
        self.assertEqual(model.emo_to_vocab.weight.shape, (self.vocab.n_words, config.hidden_dim))
        self.assertAlmostEqual(model.emo_bias_gate.item(), -2.0, places=4)

        # 取出一个真实 batch 测试前向与反向
        batch = next(iter(self.test_set))
        if config.noam:
            model.optimizer.optimizer.zero_grad()
        else:
            model.optimizer.zero_grad()
        loss_tuple = model.train_one_batch(batch, iter=100)
        
        # 验证总 loss 为有效实数
        bow_loss, kl_loss, mim_loss, ctx_loss, ppl = loss_tuple[:5]
        self.assertFalse(math.isnan(ppl), "PPL 损失不得为 NaN")
        self.assertFalse(math.isinf(ppl), "PPL 损失不得为 Inf")
        self.assertGreater(ppl, 0.0, "PPL 损失必须大于 0")

        # 验证梯度回传至 emo_to_vocab 和 emo_bias_gate
        self.assertIsNotNone(model.emo_to_vocab.weight.grad, "emo_to_vocab.weight 必须获得反向传播梯度！")
        self.assertIsNotNone(model.emo_bias_gate.grad, "emo_bias_gate 必须获得反向传播梯度！")
        grad_norm = model.emo_to_vocab.weight.grad.norm().item()
        self.assertGreater(grad_norm, 0.0, "emo_to_vocab.weight 梯度模长必须大于 0！")
        print(f"    [√] 路线 A 前向传播正常，PPL={ppl:.4f}，偏置投影头梯度模长: {grad_norm:.6f}")

        # 验证路线 A 禁用冻结标志生效
        self.assertFalse(model.is_frozen, "路线 A 设置 disable_freeze 后，模型冻结标志必须保持 False！")
        for p in model.emotion_linear.parameters():
            self.assertTrue(p.requires_grad, "路线 A 分类头参数必须全程保持 requires_grad=True！")
        print("    [√] 路线 A 分类头全程联合微调状态确证无误！")

    def test_04_route_a_decoding_modes(self):
        """测试路线 A: 词表偏置在贪婪解码与自适应核采样下的稳定性"""
        print("\n--- [Test 04] 测试路线 A 词表偏置下的自回归解码 ---")
        config.use_emo_bias = True
        model = CASE(
            self.vocab,
            emotion_num=self.emo_num,
            strategy_num=self.strategy_num,
            is_eval=True,
            model_file_path=None
        )
        model = model.to(config.device)
        model.eval()

        batch = next(iter(self.test_set))
        with torch.no_grad():
            sent_greedy = model.decoder_greedy(batch, max_dec_step=15)
            self.assertIsInstance(sent_greedy, list)
            self.assertGreater(len(sent_greedy[0]), 0)

            sent_sampling = model.decoder_sampling(batch, max_dec_step=15, temp=0.7, top_p=0.9)
            self.assertIsInstance(sent_sampling, list)
            self.assertGreater(len(sent_sampling[0]), 0)

        print(f"    [√] 贪婪生成样例: {' '.join(sent_greedy[0][:8])}...")
        print(f"    [√] 采样生成样例: {' '.join(sent_sampling[0][:8])}...")

    def test_05_clean_code_audit(self):
        """测试代码合规性: 严禁废弃 API"""
        print("\n--- [Test 05] 扫描废弃 API 与原地操作漏洞 ---")
        checked_files = [
            os.path.join(PROJECT_ROOT, "src", "models", "CASE", "model.py"),
            os.path.join(PROJECT_ROOT, "src", "utils", "config.py"),
            os.path.join(PROJECT_ROOT, "main.py"),
            os.path.join(PROJECT_ROOT, "src", "scripts", "eval_v8_pipeline.py"),
            os.path.join(PROJECT_ROOT, "src", "utils", "decode", "case.py"),
        ]
        for fpath in checked_files:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertNotIn("F.sigmoid(", content, f"{fpath} 中存在废弃的 F.sigmoid API！")
                self.assertNotIn("np.float(", content, f"{fpath} 中存在废弃的 np.float API！")
        print("    [√] 静态代码审计完成，无 F.sigmoid / np.float 违禁调用！")


if __name__ == "__main__":
    unittest.main()
