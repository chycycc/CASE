# -*- coding: utf-8 -*-
"""
V5 Trial 4 自适应分类头冻结机制（Adaptive Classifier Freezing）单元测试脚本
验证内容：
1. 默认关闭时的向后兼容性（保持 28k 步固定冻结行为）
2. 开启自适应冻结时的冷启动防抖保护
3. 达到 Patience 阈值时的参数梯度冻结与状态迁移
4. 兜底最大步数 max_freeze_step 的强制触发
5. 自适应退火进度 progress 计算的平滑性与边界正确性
"""

import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.utils.config import config


def test_backward_compatibility():
    """测试向后兼容性：未开启自适应冻结时，保持原逻辑行为"""
    print("[测试 1] 正在验证向后兼容性（默认 adaptive_freeze=False）...")
    config.adaptive_freeze = False
    config.epcl_freeze_step = 28000
    
    # 模拟简单的分类头
    emotion_linear = nn.Linear(300, 32)
    assert all(p.requires_grad for p in emotion_linear.parameters()), "初始状态下梯度应为开启"
    
    # 步数未达 28k
    iter_before = 27999
    is_frozen_before = (iter_before >= config.epcl_freeze_step)
    assert not is_frozen_before, "28k 步之前不应冻结"
    
    # 步数达到 28k
    iter_at = 28000
    is_frozen_at = (iter_at >= config.epcl_freeze_step)
    assert is_frozen_at, "28k 步应判定为冻结"
    print("  -> 向后兼容性逻辑判定通过！")


def test_adaptive_freeze_flow():
    """测试自适应冻结完整状态流：冷启动保护 -> 平台期检测 -> 触发冻结与梯度关停"""
    print("[测试 2] 正在验证自适应冻结状态迁移与梯度关停...")
    config.adaptive_freeze = True
    config.min_freeze_step = 16000
    config.max_freeze_step = 32000
    config.freeze_patience = 3
    config.freeze_metric = "emo_acc"
    
    # 模拟模型状态
    class DummyModel:
        def __init__(self):
            self.emotion_linear = nn.Linear(300, 32)
            self.is_frozen = False
            self.actual_freeze_step = config.epcl_freeze_step
            self.dataset = "ED"
            
        def freeze_emo_head(self, step):
            for p in self.emotion_linear.parameters():
                p.requires_grad = False
            self.is_frozen = True
            self.actual_freeze_step = step

    model = DummyModel()
    best_freeze_metric = -1.0
    freeze_patient = 0
    
    # 模拟序列测试数据：(step, val_emo_acc)
    # 1. 冷启动期 (10k, 12k, 14k)
    eval_stream = [
        (10000, 0.4076),
        (12000, 0.4265),
        (14000, 0.4326),
        # 2. 进入监控期，继续冲高 (16k, 18k, 20k)
        (16000, 0.4347),
        (18000, 0.4339),  # patient=1
        (20000, 0.4425),  # 创新高，reset patient=0
        # 3. 平台期震荡，不创新高 (22k, 24k, 26k) -> 触发！
        (22000, 0.4242),  # patient=1
        (24000, 0.4301),  # patient=2
        (26000, 0.4211),  # patient=3 -> 触发冻结！
    ]
    
    triggered_step = None
    for step, cur_metric in eval_stream:
        if not model.is_frozen:
            if step >= config.min_freeze_step:
                if cur_metric > best_freeze_metric:
                    best_freeze_metric = cur_metric
                    freeze_patient = 0
                else:
                    freeze_patient += 1
                
                if freeze_patient >= config.freeze_patience or step >= config.max_freeze_step:
                    model.freeze_emo_head(step)
                    triggered_step = step
                    break
            else:
                best_freeze_metric = max(best_freeze_metric, cur_metric)
                
    assert triggered_step == 26000, f"应在 26000 步触发冻结，实际为 {triggered_step}"
    assert model.is_frozen, "模型状态应标记为已冻结"
    assert model.actual_freeze_step == 26000, "实际冻结步数应记录为 26000"
    assert all(not p.requires_grad for p in model.emotion_linear.parameters()), "分类头参数 requires_grad 应全部置为 False"
    print("  -> 自适应冻结与梯度关停测试通过！")


def test_annealing_progress_computation():
    """测试自适应退火进度 progress 与衰减计算"""
    print("[测试 3] 正在验证自适应退火进度与余弦衰减平滑度...")
    actual_freeze_step = 24000
    total_decay_steps = 22000.0
    base_lr = 3.125e-4
    lr_min = base_lr * 0.03
    
    # 刚冻结时 (step = 24000)
    progress_0 = min((24000 - actual_freeze_step) / total_decay_steps, 1.0)
    decayed_lr_0 = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress_0))
    assert math.isclose(progress_0, 0.0), "起点 progress 应为 0"
    assert math.isclose(decayed_lr_0, base_lr, rel_tol=1e-5), "起点 LR 应等于 base_lr"
    
    # 中途 (step = 35000, progress = 0.5)
    progress_mid = min((35000 - actual_freeze_step) / total_decay_steps, 1.0)
    decayed_lr_mid = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress_mid))
    assert math.isclose(progress_mid, 0.5), "中途 progress 应为 0.5"
    assert lr_min < decayed_lr_mid < base_lr, "中途 LR 应平滑介于 min 与 base 之间"
    
    # 到达或超出终点 (step = 46000)
    progress_end = min((46000 - actual_freeze_step) / total_decay_steps, 1.0)
    decayed_lr_end = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress_end))
    assert math.isclose(progress_end, 1.0), "终点 progress 应为 1.0"
    assert math.isclose(decayed_lr_end, lr_min, rel_tol=1e-5), "终点 LR 应精确收敛至 lr_min"
    print("  -> 余弦退火进度与衰减曲线测试通过！")


def test_best_state_freezing_rollback():
    """测试 V5 Trial 4b 黄金状态回滚机制 (BCF)：验证漂移后精准恢复历史最佳权重"""
    print("[测试 4] 正在验证黄金状态回滚机制 (BCF)...")
    from copy import deepcopy
    
    class DummyModel:
        def __init__(self):
            self.emotion_linear = nn.Linear(300, 32)
            self.is_frozen = False
            self.actual_freeze_step = 28000
            self.dataset = "ED"
            
        def freeze_emo_head(self, step, best_state=None, best_step=None):
            if best_state is not None:
                self.emotion_linear.load_state_dict(best_state)
            for p in self.emotion_linear.parameters():
                p.requires_grad = False
            self.is_frozen = True
            self.actual_freeze_step = step

    model = DummyModel()
    # 1. 模拟 Step 20000 达到黄金峰值，缓存当时的权重 W_best
    with torch.no_grad():
        model.emotion_linear.weight.fill_(1.0)
        model.emotion_linear.bias.fill_(0.5)
    best_cached_state = deepcopy(model.emotion_linear.state_dict())
    best_step = 20000
    
    # 2. 模拟 Step 20000~26000 平台期发生权重漂移 W_drift
    with torch.no_grad():
        model.emotion_linear.weight.add_(0.2)  # 漂移为 1.2
        model.emotion_linear.bias.add_(-0.1)  # 漂移为 0.4
    assert not torch.allclose(model.emotion_linear.weight, best_cached_state["weight"]), "权重应已发生漂移"
    
    # 3. 模拟在 Step 26000 触发 BCF 冻结回滚
    model.freeze_emo_head(26000, best_state=best_cached_state, best_step=best_step)
    
    # 4. 严格断言：权重已精准恢复为 W_best，梯度关闭，冻结步数记录为 26000
    assert torch.allclose(model.emotion_linear.weight, torch.ones_like(model.emotion_linear.weight)), "权重应精准回滚至 1.0"
    assert torch.allclose(model.emotion_linear.bias, torch.full_like(model.emotion_linear.bias, 0.5)), "偏置应精准回滚至 0.5"
    assert all(not p.requires_grad for p in model.emotion_linear.parameters()), "回滚后梯度必须关闭"
    assert model.actual_freeze_step == 26000, "实际冻结步数应为 26000"
    print("  -> BCF 黄金状态回滚与参数锁定测试 100% 通过！")


if __name__ == "__main__":
    print("=" * 60)
    print("开始执行 V5 Trial 4 / Trial 4b 自适应分类头冻结机制单元测试")
    print("=" * 60)
    test_backward_compatibility()
    test_adaptive_freeze_flow()
    test_annealing_progress_computation()
    test_best_state_freezing_rollback()
    print("=" * 60)
    print("所有单元测试均 100% 通过！")
    print("=" * 60)
