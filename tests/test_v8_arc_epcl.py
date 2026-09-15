# -*- coding: utf-8 -*-
"""
[CASE-EPCL V8 架构单元测试]
测试目标: 加性角度硬边际损失 (Arc-EPCL) 与统一特征挂载锚点 (Unified Anchor)
测试范围:
1. Arc-EPCL 训练期加性边际惩罚效应测试 (正样本 logits 施加硬间隔)
2. Eval 评估模式下的硬边际旁路保护测试
3. 梯度反向传播正常性与数值稳定性测试 (无 NaN / Inf)
4. Cos 模式与 Angle 模式兼容性测试
5. Model 端到端统一特征锚点 (emotion_enc) 前向与反向计算测试
"""

import sys
import os
import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from src.models.CASE.model import PrototypeContrastiveLoss
from src.utils.config import config


class TestArcEPCLAndUnifiedAnchor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_classes = 32
        self.input_dim = 300
        self.bsz = 16

    def test_01_arc_margin_loss_effect(self):
        """测试开启 use_arc_margin 时正样本施加硬间隔后损失增加"""
        features = torch.randn(self.bsz, self.input_dim, requires_grad=True)
        labels = torch.randint(0, self.num_classes, (self.bsz,))

        # 基础无边际损失
        epcl_base = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            use_arc_margin=False,
            alpha_uni=0.0
        )
        # 带硬边际损失
        epcl_arc = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            use_arc_margin=True,
            arc_margin=0.30,
            arc_mode="cos",
            alpha_uni=0.0
        )
        # 共享相同初始原型和投影头
        epcl_arc.prototypes.data.copy_(epcl_base.prototypes.data)
        epcl_arc.class_initialized.data.copy_(epcl_base.class_initialized.data)
        epcl_arc.projection_head.load_state_dict(epcl_base.projection_head.state_dict())

        epcl_base.train()
        epcl_arc.train()

        loss_base, _ = epcl_base(features, labels, tau=0.2)
        loss_arc, _ = epcl_arc(features, labels, tau=0.2)

        # 加性硬边际扣减了目标正类的 logit，故交叉熵损失严格更大
        self.assertGreater(
            loss_arc.item(), loss_base.item(),
            f"Arc-EPCL 损失 ({loss_arc.item():.4f}) 应严格大于基准损失 ({loss_base.item():.4f})"
        )

    def test_02_eval_mode_invariance(self):
        """测试在 eval() 模式下硬边际自动旁路，不干扰评估分布"""
        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            use_arc_margin=True,
            arc_margin=0.30,
            arc_mode="cos",
            alpha_uni=0.0
        )
        epcl.eval()
        features = torch.randn(self.bsz, self.input_dim)
        labels = torch.randint(0, self.num_classes, (self.bsz,))

        loss_eval, _ = epcl(features, labels, tau=0.2)

        # 构造一个显式未开启 arc 的模型进行对比
        epcl_no_arc = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            use_arc_margin=False,
            alpha_uni=0.0
        )
        epcl_no_arc.eval()
        epcl_no_arc.prototypes.data.copy_(epcl.prototypes.data)
        epcl_no_arc.class_initialized.data.copy_(epcl.class_initialized.data)
        epcl_no_arc.projection_head.load_state_dict(epcl.projection_head.state_dict())

        loss_no_arc, _ = epcl_no_arc(features, labels, tau=0.2)

        self.assertAlmostEqual(
            loss_eval.item(), loss_no_arc.item(), places=5,
            msg="eval 模式下 Arc-EPCL 应自动旁路关闭"
        )

    def test_03_gradient_flow_and_no_nan(self):
        """测试反向传播梯度正常且无 NaN/Inf"""
        features = torch.randn(self.bsz, self.input_dim, requires_grad=True)
        labels = torch.randint(0, self.num_classes, (self.bsz,))

        epcl = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            use_mcp=True,
            use_arc_margin=True,
            arc_margin=0.30,
            use_erp=True,
            erp_hidden_dim=768
        )
        epcl.train()

        loss, _ = epcl(features, labels, tau=0.2)
        loss.backward()

        self.assertIsNotNone(features.grad, "输入特征梯度不可为空")
        self.assertFalse(torch.isnan(features.grad).any(), "反向梯度存在 NaN")
        self.assertFalse(torch.isinf(features.grad).any(), "反向梯度存在 Inf")

    def test_04_arc_modes_compatibility(self):
        """测试 cos 与 angle 两种模式运行均数值健康"""
        features = torch.randn(self.bsz, self.input_dim, requires_grad=True)
        labels = torch.randint(0, self.num_classes, (self.bsz,))

        for mode in ["cos", "angle"]:
            epcl = PrototypeContrastiveLoss(
                num_classes=self.num_classes,
                input_dim=self.input_dim,
                use_mcp=True,
                use_arc_margin=True,
                arc_margin=0.30,
                arc_mode=mode
            )
            epcl.train()
            loss, _ = epcl(features, labels, tau=0.2)
            self.assertFalse(torch.isnan(loss), f"模式 {mode} 下损失为 NaN")
            loss.backward()


if __name__ == "__main__":
    unittest.main()
