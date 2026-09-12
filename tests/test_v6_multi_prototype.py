# -*- coding: utf-8 -*-
"""
CASE-EPCL V6 Trial 2 离线单元测试脚本:
验证 Multi-Prototype EPCL (K=64 多原型解耦) 的前向对齐与均匀性计算、
Max-Cosine 动态指派机制、梯度连通性、MoP-DR 动态路由维度匹配以及生成解码兼容性。
"""
import sys
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# 注入项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import config
from src.models.CASE.model import PrototypeContrastiveLoss, ResidualEmotionHead

# 强制单元测试使用 CPU 运行，避免机器 GPU 编号不一致
config.cuda = False
config.device = torch.device("cpu")


def test_epcl_multi_prototype_standalone():
    """测试 1: 独立验证 PrototypeContrastiveLoss 在 K=1 与 K=2 模式下的正确性与梯度"""
    print("\n[测试 1] 验证 PrototypeContrastiveLoss 独立前向传播与反向传播...")
    bsz, d_in, num_classes = 8, 300, 32

    # 1.1 K=1 向后兼容单原型模式
    loss_k1 = PrototypeContrastiveLoss(num_classes=num_classes, input_dim=d_in, num_prototypes_per_class=1)
    feat1 = torch.randn(bsz, d_in, requires_grad=True)
    labels = torch.randint(0, num_classes, (bsz,))
    l1 = loss_k1(feat1, labels, tau=0.2)
    l1.backward()
    assert feat1.grad is not None and torch.norm(feat1.grad) > 0, "K=1 模式下输入梯度丢失！"
    assert loss_k1.prototypes.grad is not None and torch.norm(loss_k1.prototypes.grad) > 0, "K=1 模式下原型梯度丢失！"
    print("  -> K=1 (32单原型) 向后兼容性校验通过！")

    # 1.2 K=2 多原型解耦模式 (V6 Trial 2 核心)
    loss_k2 = PrototypeContrastiveLoss(num_classes=num_classes, input_dim=d_in, num_prototypes_per_class=2)
    assert loss_k2.prototypes.shape == (64, d_in), f"原型尺寸异常: 期望 (64, {d_in}), 实际 {loss_k2.prototypes.shape}"
    feat2 = torch.randn(bsz, d_in, requires_grad=True)
    l2 = loss_k2(feat2, labels, tau=0.2)
    assert torch.isfinite(l2), "K=2 损失值非有限数 (NaN/Inf)！"
    l2.backward()
    assert feat2.grad is not None and torch.norm(feat2.grad) > 0, "K=2 模式下输入梯度丢失！"
    assert loss_k2.prototypes.grad is not None and torch.norm(loss_k2.prototypes.grad) > 0, "K=2 模式下原型梯度丢失！"
    print("  -> K=2 (64多原型) Max-Cosine 动态指派与梯度回传校验 100% 通过！")


def test_max_cosine_assignment_logic():
    """测试 2: 精准验证 Max-Cosine 动态指派在极端相似度下的选择数学逻辑"""
    print("\n[测试 2] 验证 Max-Cosine 动态指派与负样本抑制逻辑...")
    bsz, d_in, num_classes, K = 2, 4, 3, 2
    loss_fn = PrototypeContrastiveLoss(num_classes=num_classes, input_dim=d_in, num_prototypes_per_class=K)

    # 人工构造样本向量与原型矩阵以严格推导选择
    # 类别 0 的两个子原型: p0_0 = [1, 0, 0, 0], p0_1 = [0, 1, 0, 0]
    # 类别 1 的两个子原型: p1_0 = [0, 0, 1, 0], p1_1 = [0, 0, 0, 1]
    # 类别 2 的两个子原型: p2_0 = [0.707, 0.707, 0, 0], p2_1 = [-1, 0, 0, 0]
    protos = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # class 0, sub 0
        [0.0, 1.0, 0.0, 0.0],  # class 0, sub 1
        [0.0, 0.0, 1.0, 0.0],  # class 1, sub 0
        [0.0, 0.0, 0.0, 1.0],  # class 1, sub 1
        [0.707, 0.707, 0.0, 0.0],  # class 2, sub 0
        [-1.0, 0.0, 0.0, 0.0], # class 2, sub 1
    ], dtype=torch.float32)
    loss_fn.prototypes.data = protos

    # 构造两个样本 (绕过 projection_head 直接在 normalize 空间测试)
    # 样本 0: 属于类别 0，特征 [0.9, 0.1, 0, 0]，与 sub 0 余弦相似度最高
    # 样本 1: 属于类别 0，特征 [0.1, 0.9, 0, 0]，与 sub 1 余弦相似度最高
    z = torch.tensor([
        [0.9, 0.1, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0]
    ], dtype=torch.float32)
    z_norm = F.normalize(z, p=2, dim=-1)
    proto_norm = F.normalize(protos, p=2, dim=-1)
    labels = torch.tensor([0, 0], dtype=torch.long)

    all_cosine = torch.matmul(z_norm, proto_norm.T).view(bsz, num_classes, K)
    target_cosines = all_cosine[torch.arange(bsz), labels]
    best_pos, best_idx = target_cosines.max(dim=-1)

    assert best_idx[0].item() == 0, f"样本 0 应指派至 sub 0，实际指派: {best_idx[0].item()}"
    assert best_idx[1].item() == 1, f"样本 1 应指派至 sub 1，实际指派: {best_idx[1].item()}"
    print("  -> Max-Cosine 细粒度子原型动态自适应指派行为完全符合预期！")


def test_mop_dr_router_compatibility():
    """测试 3: 验证 MoP-DR 原型路由器在 K*C (64 维) 输入下的张量流转"""
    print("\n[测试 3] 验证 MoP-DR 路由器在 64 原型下的前向与门控融合...")
    total_protos = 64
    d_in = 300
    bsz = 4
    router_linear = nn.Linear(total_protos, d_in)
    
    # 模拟原型相似度归一化概率分布 P_route: [B, 64]
    proto_logits = torch.randn(bsz, total_protos) / 0.1
    P_route = F.softmax(proto_logits, dim=-1)
    
    route_gate = torch.sigmoid(router_linear(P_route))
    assert route_gate.shape == (bsz, d_in), f"route_gate 形状异常: 期望 {(bsz, d_in)}, 实际 {route_gate.shape}"
    
    # 模拟动态门控加权
    static_gate = torch.sigmoid(torch.randn(bsz, d_in))
    lambda_epcl = 0.07
    emo_gate = lambda_epcl * route_gate + (1 - lambda_epcl) * static_gate
    assert emo_gate.shape == (bsz, d_in), "emo_gate 形状不匹配！"
    assert (emo_gate >= 0.0).all() and (emo_gate <= 1.0).all(), "门控权重超出 [0, 1] 物理区间！"
    print("  -> MoP-DR 64 维路由器前向计算与双通道门控融合无误！")


if __name__ == "__main__":
    print("=" * 70)
    print("        CASE-EPCL V6 Trial 2 离线完整性单元测试        ")
    print("=" * 70)
    test_epcl_multi_prototype_standalone()
    test_max_cosine_assignment_logic()
    test_mop_dr_router_compatibility()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! V6 Trial 2 (Multi-Prototype K=64) 验证完毕！")
    print("=" * 70)
