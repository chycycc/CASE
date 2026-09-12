# -*- coding: utf-8 -*-
"""
CASE-EPCL V6 Trial 3 离线单元测试脚本:
验证 Prototype-Conditioned Attention Module (PCAM) 架构:
1. PCAM 模块独立前向与反向传播 (尺寸匹配、梯度回传至解码隐状态与原型矩阵);
2. 平滑暖启动门控机制验证 (初始负偏置阻断剧烈扰动，保护自回归流畅度);
3. 自回归单步解码 (seq_len=1) 与多批次极端尺寸兼容性;
4. CASE 主模型集成与 apply_pcam 开关向后兼容性验证。
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 注入项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import config
from src.models.CASE.model import PrototypeConditionedAttentionModule, CASE

# 强制单元测试使用 CPU 运行
config.cuda = False
config.device = torch.device("cpu")


def test_pcam_standalone_forward_and_backward():
    """测试 1: 独立验证 PCAM 模块前向与反向传播"""
    print("\n[测试 1] 验证 PCAM 模块独立前向与反向传播...")
    bsz, seq_len, d_model = 4, 15, 300
    k_protos = 64

    pcam = PrototypeConditionedAttentionModule(
        d_model=d_model,
        num_heads=2,
        dropout=0.1,
        gate_bias=-1.0
    )

    dec_output = torch.randn(bsz, seq_len, d_model, requires_grad=True)
    prototypes = torch.randn(k_protos, d_model, requires_grad=True)

    out = pcam(dec_output, prototypes)

    # 1. 验证尺寸
    assert out.shape == (bsz, seq_len, d_model), f"PCAM 输出尺寸异常: 期望 {(bsz, seq_len, d_model)}, 实际 {out.shape}"
    assert torch.isfinite(out).all(), "PCAM 输出包含 NaN 或 Inf！"

    # 2. 验证反向传播与梯度连通性
    loss = out.sum()
    loss.backward()

    assert dec_output.grad is not None and torch.norm(dec_output.grad) > 0, "dec_output 梯度丢失！"
    assert prototypes.grad is not None and torch.norm(prototypes.grad) > 0, "prototypes 梯度未回传至原型矩阵！"
    assert pcam.q_proj.weight.grad is not None and torch.norm(pcam.q_proj.weight.grad) > 0, "PCAM q_proj 梯度丢失！"
    assert pcam.gate_linear.weight.grad is not None and torch.norm(pcam.gate_linear.weight.grad) > 0, "门控网络梯度丢失！"

    print("  -> PCAM 尺寸、数值稳定性与全路径梯度连通性 100% 校验通过！")


def test_pcam_smooth_warmstart_gating():
    """测试 2: 验证 PCAM 平滑暖启动门控初始值分布"""
    print("\n[测试 2] 验证平滑暖启动门控特性...")
    d_model = 300
    gate_bias = -1.0
    pcam = PrototypeConditionedAttentionModule(
        d_model=d_model,
        num_heads=2,
        dropout=0.0,
        gate_bias=gate_bias
    )
    pcam.eval()

    bsz, seq_len = 8, 20
    k_protos = 64
    dec_output = torch.randn(bsz, seq_len, d_model)
    prototypes = torch.randn(k_protos, d_model)

    with torch.no_grad():
        # 获取门控激活值
        q = pcam.q_proj(dec_output) * pcam.scaling
        k = pcam.k_proj(prototypes)
        v = pcam.v_proj(prototypes)
        q = q.view(bsz, seq_len, pcam.num_heads, pcam.head_dim).transpose(1, 2)
        k = k.view(k_protos, pcam.num_heads, pcam.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
        v = v.view(k_protos, pcam.num_heads, pcam.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_weights, dim=-1)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, pcam.d_model)
        h_proto = pcam.out_proj(context)

        gate_input = torch.cat([dec_output, h_proto], dim=-1)
        gate = torch.sigmoid(pcam.gate_linear(gate_input))

    mean_gate = gate.mean().item()
    expected_approx = 1.0 / (1.0 + 2.718281828 ** (-gate_bias))  # sigmoid(-1.0) approx 0.2689
    print(f"  -> 实测初始门控均值: {mean_gate:.4f} (期望理论基准 ≈ {expected_approx:.4f})")
    assert 0.15 < mean_gate < 0.40, f"门控初始值脱离平滑暖启动安全区: {mean_gate}"
    print("  -> 平滑暖启动门控验证通过！")


def test_pcam_autoregressive_step():
    """测试 3: 验证自回归生成模式下 seq_len=1 的单步解码表现"""
    print("\n[测试 3] 验证 seq_len=1 单步解码 (Greedy/Beam Search) 兼容性...")
    pcam = PrototypeConditionedAttentionModule(d_model=300, num_heads=2, dropout=0.0)
    pcam.eval()

    # 模拟 Beam Search 时单步 batch=4 (n_active * n_bm)
    dec_step_output = torch.randn(4, 1, 300)
    prototypes = torch.randn(64, 300)

    with torch.no_grad():
        out_step = pcam(dec_step_output, prototypes)

    assert out_step.shape == (4, 1, 300), f"单步解码输出尺寸异常: {out_step.shape}"
    assert torch.isfinite(out_step).all(), "单步解码出现 NaN！"
    print("  -> seq_len=1 单步自回归解码 100% 兼容通过！")


def test_case_model_apply_pcam_integration():
    """测试 4: 验证 CASE 主模型的 apply_pcam 开关与向后兼容性"""
    print("\n[测试 4] 验证 CASE 主模型集成与 apply_pcam 开关...")

    class MockVocab:
        def __init__(self):
            self.n_words = 1000
            self.word2index = {}
            self.index2word = {i: f"word_{i}" for i in range(1000)}

    vocab = MockVocab()

    # 4.1 测试开启 PCAM 状态
    config.use_pcam = True
    config.pcam_heads = 2
    config.num_prototypes_per_class = 2
    config.emotion_head_type = "residual_mlp"
    model_pcam = CASE(vocab, emotion_num=32, strategy_num=8)

    assert hasattr(model_pcam, "pcam"), "CASE 模型中缺少 pcam 模块！"
    assert model_pcam.use_pcam is True, "CASE 模型 use_pcam 状态未生效！"

    dummy_dec = torch.randn(2, 5, 300)
    enhanced = model_pcam.apply_pcam(dummy_dec)
    assert enhanced.shape == (2, 5, 300), f"apply_pcam 输出尺寸异常: {enhanced.shape}"
    assert not torch.allclose(enhanced, dummy_dec), "开启 PCAM 时 apply_pcam 应产生原型特征强化！"
    print("  -> PCAM 开启状态下 apply_pcam 正确强化隐状态！")

    # 4.2 测试关闭 PCAM 状态 (向后兼容性)
    config.use_pcam = False
    model_legacy = CASE(vocab, emotion_num=32, strategy_num=8)
    assert not hasattr(model_legacy, "pcam"), "关闭 PCAM 时不应实例化 pcam 模块！"
    raw_dec = torch.randn(2, 5, 300)
    passthrough = model_legacy.apply_pcam(raw_dec)
    assert torch.equal(passthrough, raw_dec), "关闭 PCAM 时 apply_pcam 必须严格原样透传！"
    print("  -> PCAM 关闭状态下 apply_pcam 严格原样透传，向后兼容 100% 通过！")


if __name__ == "__main__":
    print("=" * 60)
    print("CASE-EPCL V6 Trial 3 (PCAM) 单元测试套件启动")
    print("=" * 60)

    test_pcam_standalone_forward_and_backward()
    test_pcam_smooth_warmstart_gating()
    test_pcam_autoregressive_step()
    test_case_model_apply_pcam_integration()

    print("\n" + "=" * 60)
    print("[PASS] 所有 PCAM 单元测试全部通过 (4/4 PASS)！")
    print("=" * 60)
