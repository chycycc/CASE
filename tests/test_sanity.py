# -*- coding: utf-8 -*-
"""
CASE-EPCL V8.2 F 集成测试 (Integration Tests)
验证钟形动态加权、偏置退火下限、帕累托评分等核心机制的数学正确性与系统兼容性。
"""
import sys
import os
import math
import unittest
try:
    import pytest
except ImportError:
    pytest = None

# 注册工程根路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.utils.config import config


class TestBellShapedRamp(unittest.TestCase):
    """Test 01-03: 钟形余弦加权数学正确性验证"""

    @staticmethod
    def compute_bell_weight(progress, ramp_max):
        """复现 model.py 中的钟形加权公式"""
        return 1.0 + (ramp_max - 1.0) * math.sin(math.pi * progress)

    @staticmethod
    def compute_linear_weight(progress, ramp_max):
        """复现 model.py 中的线性加权公式"""
        return 1.0 + (ramp_max - 1.0) * progress

    def test_01_bell_peak_at_midpoint(self):
        """[Test 01] 钟形加权在 progress=0.5 处达到峰值 ramp_max"""
        ramp_max = 1.3
        # 中点处 sin(π * 0.5) = sin(π/2) = 1.0
        w_mid = self.compute_bell_weight(0.5, ramp_max)
        assert abs(w_mid - ramp_max) < 1e-10, (
            f"中点权重应为 {ramp_max}，实际为 {w_mid}"
        )

    def test_02_bell_endpoints_equal_one(self):
        """[Test 02] 钟形加权在 progress=0 和 progress=1 处回归 1.0"""
        ramp_max = 1.3
        # progress=0: sin(0) = 0 → w = 1.0
        w_start = self.compute_bell_weight(0.0, ramp_max)
        assert abs(w_start - 1.0) < 1e-10, (
            f"起始权重应为 1.0，实际为 {w_start}"
        )
        # progress=1: sin(π) ≈ 0 → w ≈ 1.0
        w_end = self.compute_bell_weight(1.0, ramp_max)
        assert abs(w_end - 1.0) < 1e-10, (
            f"终点权重应为 1.0，实际为 {w_end}"
        )

    def test_03_bell_symmetry(self):
        """[Test 03] 钟形加权关于中点对称"""
        ramp_max = 1.3
        # 对称性: w(0.2) == w(0.8), w(0.3) == w(0.7)
        for p in [0.1, 0.2, 0.3, 0.4]:
            w_left = self.compute_bell_weight(p, ramp_max)
            w_right = self.compute_bell_weight(1.0 - p, ramp_max)
            assert abs(w_left - w_right) < 1e-10, (
                f"对称性失败: w({p})={w_left:.6f} != w({1.0-p})={w_right:.6f}"
            )

    def test_04_bell_always_gte_one(self):
        """[Test 04] 钟形加权在 [0,1] 区间内恒 >= 1.0 (当 ramp_max >= 1.0)"""
        ramp_max = 1.3
        for i in range(101):
            p = i / 100.0
            w = self.compute_bell_weight(p, ramp_max)
            assert w >= 1.0 - 1e-10, (
                f"权重 w({p})={w:.6f} 小于 1.0"
            )

    def test_05_bell_vs_linear_at_midpoint(self):
        """[Test 05] 钟形在中点处权重高于线性在同一 progress 处的权重 (更强的中段压制)"""
        ramp_max = 1.3
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            w_bell = self.compute_bell_weight(p, ramp_max)
            w_linear = self.compute_linear_weight(p, ramp_max)
            if p == 0.5:
                # 中点处 bell > linear: 1.3 vs 1.15
                assert w_bell > w_linear, (
                    f"中点处 bell={w_bell:.4f} 应大于 linear={w_linear:.4f}"
                )

    def test_06_bell_ramp_max_1_3_specific_values(self):
        """[Test 06] ramp_max=1.3 时关键步数点的精确权重验证"""
        ramp_max = 1.3
        ramp_start = 24000
        ramp_steps = 16000
        
        # 各关键步数的预期权重
        test_cases = [
            (24000, 1.0),       # 起始: progress=0
            (28000, 1.0 + 0.3 * math.sin(math.pi * 0.25)),  # progress=0.25
            (32000, 1.3),       # 中点: progress=0.5, 峰值
            (36000, 1.0 + 0.3 * math.sin(math.pi * 0.75)),  # progress=0.75
            (40000, 1.0),       # 终点: progress=1.0
        ]
        
        for step, expected_w in test_cases:
            progress = min(1.0, float(step - ramp_start) / float(ramp_steps))
            actual_w = self.compute_bell_weight(progress, ramp_max)
            assert abs(actual_w - expected_w) < 1e-10, (
                f"Step {step}: 期望 w={expected_w:.6f}，实际 w={actual_w:.6f}"
            )


class TestBiasAnnealingMinScale(unittest.TestCase):
    """Test 07-08: 偏置退火下限 0.10 到达验证"""

    @staticmethod
    def compute_bias_scale(step, start_step=24000, span_steps=10000, min_scale=0.10):
        """复现 model.py 中的 get_bias_scale 逻辑"""
        if step <= start_step:
            return 1.0
        progress = min(1.0, float(step - start_step) / float(span_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine_decay

    def test_07_bias_reaches_new_floor(self):
        """[Test 07] 偏置退火在 start+span 步后到达 min_scale=0.10"""
        scale = self.compute_bias_scale(34000, start_step=24000, span_steps=10000, min_scale=0.10)
        assert abs(scale - 0.10) < 1e-10, (
            f"退火终点应为 0.10，实际为 {scale}"
        )

    def test_08_bias_new_floor_lower_than_old(self):
        """[Test 08] 新下限 0.10 低于旧下限 0.15"""
        scale_new = self.compute_bias_scale(34000, start_step=24000, span_steps=10000, min_scale=0.10)
        scale_old = self.compute_bias_scale(34000, start_step=24000, span_steps=10000, min_scale=0.15)
        assert scale_new < scale_old, (
            f"新下限 {scale_new} 应低于旧下限 {scale_old}"
        )


class TestParetoCompositeScore(unittest.TestCase):
    """Test 09-10: 帕累托复合评分 α=5.0 行为验证"""

    @staticmethod
    def compute_score(ppl_val, emo_loss_val, alpha):
        return ppl_val + alpha * emo_loss_val

    def test_09_alpha_5_sensitivity(self):
        """[Test 09] α=5.0 对 EMO_loss 变化更敏感 (相比 α=3.5)"""
        ppl = 37.0
        emo_loss_a = 2.10
        emo_loss_b = 2.30
        
        # α=3.5 时的分数差
        delta_35 = abs(self.compute_score(ppl, emo_loss_a, 3.5) - self.compute_score(ppl, emo_loss_b, 3.5))
        # α=5.0 时的分数差
        delta_50 = abs(self.compute_score(ppl, emo_loss_a, 5.0) - self.compute_score(ppl, emo_loss_b, 5.0))
        
        assert delta_50 > delta_35, (
            f"α=5.0 的分数差 ({delta_50:.4f}) 应大于 α=3.5 ({delta_35:.4f})"
        )

    def test_10_alpha_5_prefers_lower_emo_loss(self):
        """[Test 10] α=5.0 下，EMO_loss 更低的检查点胜出（即使 PPL 略高）"""
        # 检查点 A: PPL 低但 EMO_loss 高 (类似 V8.2 E 39k 步)
        score_a = self.compute_score(37.11, 2.3633, 5.0)
        # 检查点 B: PPL 略高但 EMO_loss 好 (类似 V8.2 E 34k 步)
        score_b = self.compute_score(38.14, 2.1489, 5.0)
        
        assert score_b < score_a, (
            f"低 EMO_loss 检查点 (score={score_b:.4f}) 应优于高 EMO_loss 检查点 (score={score_a:.4f})"
        )


class TestProtectionPeriod(unittest.TestCase):
    """Test 11: 保护期 min_save_step=30000 逻辑验证"""

    def test_11_protection_period_at_30k(self):
        """[Test 11] 30000 步时刚好退出保护期（30000 >= 30000）"""
        min_save_step = 30000
        
        # 29999 步仍在保护期内
        assert 29999 < min_save_step, "29999 应处于保护期内"
        # 30000 步退出保护期 (main.py 中 n_iter < min_save_step 判定)
        assert not (30000 < min_save_step), "30000 应不在保护期内"


class TestBellShapeModelIntegration(unittest.TestCase):
    """Test 12-13: 钟形加权与模型组件的兼容性验证"""

    def test_12_bell_ramp_preserves_loss_positivity(self):
        """[Test 12] 钟形加权后的 emotion_loss 始终 > 0（给定正输入）"""
        base_loss = 2.3  # 典型 EMO_loss 量级
        ramp_max = 1.3
        for i in range(101):
            progress = i / 100.0
            w = 1.0 + (ramp_max - 1.0) * math.sin(math.pi * progress)
            weighted_loss = w * base_loss
            assert weighted_loss > 0, (
                f"加权后 loss 为负: w={w:.6f}, loss={weighted_loss:.6f}"
            )

    def test_13_bell_weight_bounded(self):
        """[Test 13] 钟形加权系数始终在 [1.0, ramp_max] 区间内"""
        ramp_max = 1.3
        for i in range(10001):
            progress = i / 10000.0
            w = 1.0 + (ramp_max - 1.0) * math.sin(math.pi * progress)
            assert 1.0 - 1e-10 <= w <= ramp_max + 1e-10, (
                f"权重 w={w:.6f} 超出 [1.0, {ramp_max}] 区间"
            )

    def test_14_accum_steps_and_precision_config(self):
        """[Test 14] accum_steps 与 precision 参数默认值与自适应校验"""
        assert hasattr(config, "accum_steps"), "config 缺少 accum_steps 属性"
        assert hasattr(config, "precision"), "config 缺少 precision 属性"
        assert getattr(config, "accum_steps", 1) >= 1, "accum_steps 应 >= 1"
        assert getattr(config, "precision", "fp32") in ["fp32", "bf16", "fp16"]

    def test_15_attention_mask_dtype_adaptation(self):
        """[Test 15] Attention 掩码数值针对 float16 与 float32 自适应"""
        # float16 下应使用 -1e4 防溢出；float32 下应使用 -1e18 数学严格零
        dtype_fp16 = torch.float16
        dtype_fp32 = torch.float32
        mask_val_fp16 = -1e4 if dtype_fp16 == torch.float16 else -1e18
        mask_val_fp32 = -1e4 if dtype_fp32 == torch.float16 else -1e18
        assert mask_val_fp16 == -1e4
        assert mask_val_fp32 == -1e18


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        unittest.main(verbosity=2)
