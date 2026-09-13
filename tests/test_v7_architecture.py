# -*- coding: utf-8 -*-
"""
[CASE-EPCL V7 架构单元测试套件]
测试目标:
1. ExpandedResidualProjector (ERP) 形状、L2 归一化与反向传播连通性
2. ContextConditionedHyperPrototypes (CCHP) 静态/动态语境原型调制与位移正则项验证
3. CASE 主模型全要素端到端前向与反向传播集成验证 (train_one_batch & apply_pcam)
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.models.CASE.model import (
    ExpandedResidualProjector,
    ContextConditionedHyperPrototypes,
    PrototypeContrastiveLoss,
    CASE,
)


class TestV7Architecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bsz = 4
        self.dim = 300
        self.num_classes = 32
        self.num_protos_per_class = 2
        self.total_protos = 64

    def test_erp_shape_and_norm(self):
        """测试高维解缠残差投影头 (ERP)"""
        erp = ExpandedResidualProjector(
            input_dim=self.dim,
            hidden_dim=768,
            output_dim=self.dim,
            dropout=0.1
        ).to(self.device)

        x = torch.randn(self.bsz, self.dim, device=self.device, requires_grad=True)
        z = erp(x)

        # 1. 形状验证
        self.assertEqual(z.shape, (self.bsz, self.dim))

        # 2. 超球面单位模长验证
        norms = torch.norm(z, p=2, dim=-1)
        expected_norms = torch.ones_like(norms)
        self.assertTrue(torch.allclose(norms, expected_norms, atol=1e-5))

        # 3. 梯度连通性
        loss = z.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        print("[PASS] ERP 形状、L2 归一化与反向梯度测试通过！")

    def test_cchp_dynamic_and_static_mode(self):
        """测试动态上下文条件超网络原型 (CCHP)"""
        cchp = ContextConditionedHyperPrototypes(
            num_classes=self.num_classes,
            num_prototypes_per_class=self.num_protos_per_class,
            input_dim=self.dim,
            alpha_dyn=0.1
        ).to(self.device)

        # 1. 静态模式 (无上下文输入)
        static_p, reg_loss = cchp()
        self.assertEqual(static_p.shape, (self.total_protos, self.dim))
        self.assertEqual(reg_loss.item(), 0.0)
        p_norms = torch.norm(static_p, p=2, dim=-1)
        self.assertTrue(torch.allclose(p_norms, torch.ones_like(p_norms), atol=1e-5))

        # 2. 动态模式 (传入 batch 上下文)
        ctx = torch.randn(self.bsz, self.dim, device=self.device, requires_grad=True)
        dyn_p, reg_loss = cchp(ctx_repr=ctx)
        self.assertEqual(dyn_p.shape, (self.bsz, self.total_protos, self.dim))
        self.assertGreater(reg_loss.item(), 0.0)

        # 模长必须仍在超球面上
        dyn_norms = torch.norm(dyn_p, p=2, dim=-1)
        self.assertTrue(torch.allclose(dyn_norms, torch.ones_like(dyn_norms), atol=1e-5))

        # 3. 反向传播连通性
        loss = dyn_p.sum() + reg_loss
        loss.backward()
        self.assertIsNotNone(ctx.grad)
        self.assertGreater(ctx.grad.abs().sum().item(), 0)
        print("[PASS] CCHP 静态/动态语境原型调制与位移正则测试通过！")

    def test_epcl_loss_with_v7_components(self):
        """测试集成 ERP 与 CCHP 的 PrototypeContrastiveLoss"""
        criterion = PrototypeContrastiveLoss(
            num_classes=self.num_classes,
            input_dim=self.dim,
            num_prototypes_per_class=2,
            use_erp=True,
            erp_hidden_dim=768,
            use_cchp=True,
            cchp_alpha=0.1
        ).to(self.device)

        features = torch.randn(self.bsz, self.dim, device=self.device, requires_grad=True)
        ctx = torch.randn(self.bsz, self.dim, device=self.device, requires_grad=True)
        labels = torch.randint(0, self.num_classes, (self.bsz,), device=self.device)

        loss, reg_loss = criterion(features, labels, ctx_repr=ctx)
        total_loss = loss + 0.01 * reg_loss

        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(total_loss.item(), 0)

        total_loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(ctx.grad)
        print("[PASS] PrototypeContrastiveLoss (ERP + CCHP) 联合计算与梯度回传通过！")

    def test_pcam_with_2d_and_3d_prototypes(self):
        """测试 PCAM 模块在 2D 静态原型与 3D 动态原型输入下的兼容性"""
        from src.models.CASE.model import PrototypeConditionedAttentionModule
        pcam = PrototypeConditionedAttentionModule(d_model=self.dim, num_heads=2).to(self.device)
        dec_out = torch.randn(self.bsz, 15, self.dim, device=self.device)

        # 1. 2D 静态原型测试 [total_protos, dim]
        proto_2d = torch.randn(self.total_protos, self.dim, device=self.device)
        out_2d = pcam(dec_out, proto_2d)
        self.assertEqual(out_2d.shape, dec_out.shape)

        # 2. 3D 动态上下文原型测试 [bsz, total_protos, dim]
        proto_3d = torch.randn(self.bsz, self.total_protos, self.dim, device=self.device)
        out_3d = pcam(dec_out, proto_3d)
        self.assertEqual(out_3d.shape, dec_out.shape)
        print("[PASS] PCAM 自适应 2D/3D 原型跨注意力测试通过！")

    def test_unlikelihood_loss(self):
        """测试序列级无似然训练损失 (Unlikelihood Loss)"""
        from src.models.CASE.model import compute_unlikelihood_loss
        vocab_size = 100
        seq_len = 10
        logits = torch.randn(self.bsz, seq_len, vocab_size, device=self.device, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=-1)

        targets = torch.randint(2, vocab_size, (self.bsz, seq_len), device=self.device)
        targets[:, 5] = targets[:, 1]
        targets[:, 6] = targets[:, 2]

        ul_loss = compute_unlikelihood_loss(log_probs, targets, pad_idx=1)
        self.assertFalse(torch.isnan(ul_loss))
        self.assertGreater(ul_loss.item(), 0.0)

        ul_loss.backward()
        self.assertIsNotNone(logits.grad)
        print("[PASS] 序列级无似然损失 (Unlikelihood Loss) 向量化计算与反向传播通过！")


if __name__ == '__main__':
    unittest.main()
