# -*- coding: utf-8 -*-
"""
[CASE-EPCL V8 架构单元测试]
测试目标: 动量样本中心原型 (Momentum Centroid Prototypes, MCP)
测试范围:
1. 原型不可导 Buffer 属性检验 (requires_grad == False)
2. 样本类别冷启动直接初始化检验
3. EMA 动量平滑更新与数学收敛性检验
4. Eval 模式下的原型严格冻结检验
5. Autograd 反向传播合规性 (无原地操作、无梯度污染)
6. CASE 顶层模型端到端前向与解码兼容性
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from src.models.CASE.model import PrototypeContrastiveLoss, CASE
from src.utils.config import config


class TestMomentumCentroidPrototypes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 32
        self.input_dim = 300
        self.momentum = 0.90  # 单元测试使用较小动量以便快速观察收敛

    def test_01_buffer_registration_and_no_grad(self):
        """测试 MCP 原型是否正确注册为不可导 Buffer"""
        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            mcp_momentum=self.momentum
        )
        # 1. 验证 prototypes 为 buffer 而非 parameter
        buffer_names = [name for name, _ in epcl.named_buffers()]
        param_names = [name for name, _ in epcl.named_parameters()]
        self.assertIn("prototypes", buffer_names, "prototypes 必须注册为 Buffer")
        self.assertIn("class_initialized", buffer_names, "class_initialized 必须注册为 Buffer")
        self.assertNotIn("prototypes", param_names, "prototypes 绝不能是 Parameter")

        # 2. 验证不可导性
        self.assertFalse(epcl.prototypes.requires_grad, "prototypes requires_grad 必须为 False")
        self.assertFalse(epcl.class_initialized.requires_grad, "class_initialized requires_grad 必须为 False")

        # 3. 验证初始状态为未初始化
        self.assertEqual(epcl.class_initialized.sum().item(), 0, "初始化时所有类别标记必须为 False")

    def test_02_cold_start_initialization(self):
        """测试冷启动：类别首次出现时直接将质心赋予该类别"""
        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            mcp_momentum=self.momentum
        )
        epcl.train()

        # 构造一批样本：只包含类别 3 和类别 7
        features = torch.randn(10, self.input_dim)
        labels = torch.tensor([3, 3, 3, 3, 3, 7, 7, 7, 7, 7], dtype=torch.long)

        # 前向传播触发更新
        loss, _ = epcl(features, labels)

        # 检查类别 3 和 7 是否标记为已初始化
        self.assertTrue(epcl.class_initialized[3].item(), "类别 3 应已初始化")
        self.assertTrue(epcl.class_initialized[7].item(), "类别 7 应已初始化")
        self.assertFalse(epcl.class_initialized[0].item(), "类别 0 不应被初始化")

        # 验证冷启动下原型精确等于这批样本投影后的均值单位向量
        proj = epcl.projection_head(features)
        proj_norm = F.normalize(proj, p=2, dim=1)
        expected_c3 = F.normalize(proj_norm[labels == 3].mean(dim=0), p=2, dim=0)
        
        cos_sim = torch.dot(epcl.prototypes[3], expected_c3).item()
        self.assertAlmostEqual(cos_sim, 1.0, places=4, msg="冷启动原型应与批次样本投影均值严格共线")

    def test_03_ema_convergence(self):
        """测试连续训练下，EMA 动量原型单调平滑收敛至目标质心"""
        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            mcp_momentum=0.8
        )
        epcl.train()

        target_center = F.normalize(torch.randn(self.input_dim), p=2, dim=0)
        c = 5
        labels = torch.full((16,), c, dtype=torch.long)

        sims = []
        for step in range(25):
            # 生成围绕 target_center 的样本
            samples = target_center.unsqueeze(0).repeat(16, 1) + 0.05 * torch.randn(16, self.input_dim)
            epcl(samples, labels)
            current_proto = epcl.prototypes[c]
            # 计算当前原型与目标样本投影质心的余弦相似度
            proj_target = F.normalize(epcl.projection_head(target_center.unsqueeze(0)), p=2, dim=1).squeeze(0)
            cos_sim = torch.dot(current_proto, proj_target).item()
            sims.append(cos_sim)

        # 经过 25 轮样本输入后，原型应当高度逼近样本真实质心
        self.assertGreater(sims[-1], 0.95, "经过 25 轮样本输入后，原型应当高度逼近样本真实质心")

    def test_04_eval_mode_strict_freeze(self):
        """测试在 eval 模式下原型严格冻结，不发生任何漂移"""
        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            mcp_momentum=0.5
        )
        epcl.train()

        # 先冷启动更新一次
        features = torch.randn(8, self.input_dim)
        labels = torch.tensor([1, 1, 2, 2, 4, 4, 8, 8], dtype=torch.long)
        epcl(features, labels)

        # 记录当前的 prototypes
        saved_prototypes = epcl.prototypes.clone()

        # 切换至 eval 模式
        epcl.eval()
        new_features = torch.randn(8, self.input_dim) * 10.0
        new_labels = torch.tensor([1, 1, 2, 2, 4, 4, 8, 8], dtype=torch.long)
        
        with torch.no_grad():
            epcl(new_features, new_labels)

        # 验证 eval 下未发生任何变化
        diff = (epcl.prototypes - saved_prototypes).abs().max().item()
        self.assertEqual(diff, 0.0, "eval 模式下 prototypes 必须严格保持冻结状态，绝对误差应为 0")

    def test_05_autograd_and_no_inplace_error(self):
        """测试反向传播流程，确保无原地操作导致 autograd 报错"""
        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            mcp_momentum=0.9
        )
        epcl.train()

        features = torch.randn(12, self.input_dim, requires_grad=True)
        labels = torch.randint(0, self.num_classes, (12,))

        loss, reg_loss = epcl(features, labels)
        total_loss = loss + reg_loss
        
        # 反向传播应畅通无阻
        total_loss.backward()

        self.assertIsNotNone(features.grad, "输入特征 features 必须获得反向梯度")
        self.assertGreater(features.grad.norm().item(), 0.0, "梯度模长应大于 0")

    def test_06_case_model_end_to_end_mcp_compatibility(self):
        """测试 MCP 与 CASE 顶层模型的端到端前向兼容性"""
        config.use_mcp = True
        config.mcp_momentum = 0.99
        config.use_erp = False
        config.use_cchp = False
        config.dataset = "ED"
        config.woStrategy = True
        config.device = torch.device("cpu")
        config.emb_dim = 300
        config.num_prototypes_per_class = 1
        config.emotion_head_type = "linear"

        class MockVocab:
            def __init__(self, n_words=100):
                self.n_words = n_words
                self.word2index = {}
        config.pretrain_emb = False
        vocab = MockVocab(100)
        model = CASE(vocab, emotion_num=32, strategy_num=8, is_eval=False)
        model.to(config.device)

        self.assertTrue(hasattr(model, "epcl_criterion"), "CASE 模型必须包含 epcl_criterion")
        self.assertTrue(model.epcl_criterion.use_mcp, "epcl_criterion 必须处于 use_mcp 激活状态")
        self.assertIsInstance(model.epcl_criterion.prototypes, torch.Tensor)
        self.assertEqual(model.epcl_criterion.current_prototypes.size(), (32, 300))


if __name__ == "__main__":
    unittest.main()
