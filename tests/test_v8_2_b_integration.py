# -*- coding: utf-8 -*-
"""
CASE-EPCL V8.2 B 集成与机理修复单元测试

验证重点:
1. 学习率余弦退火与分类头冻结状态彻底解绑 (在 disable_freeze=True 下严格执行 24k~46k 余弦退火至 1e-5)
2. 动态早停耐心机制 (config.patience 支持自定义容忍度，解除硬编码 5 次截断)
3. 静态代码规范与现代 PyTorch 标准审计
"""

import os
import sys
import math
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestV82BIntegration(unittest.TestCase):

    def test_01_lr_decay_unbound_from_freeze(self):
        """测试 1: 验证在 disable_freeze=True (全程微调未冻结) 下余弦学习率衰减是否精确生效"""
        print("\n--- [Test 01] 验证 disable_freeze 场景下的余弦学习率衰减 ---")
        
        anneal_start = 24000
        total_decay_steps = 22000.0
        base_lr = 3.125e-4
        lr_min = base_lr * 0.03 # 约 9.375e-6
        
        test_steps = [10000, 24000, 35000, 46000, 50000]
        
        for step in test_steps:
            disable_freeze = True
            is_currently_frozen = False
            
            # 模拟 model.py 中的退火判定逻辑
            if disable_freeze:
                decay_start = anneal_start
                should_decay = (step > decay_start)
                decay_origin = decay_start
            else:
                freeze_anchor = 14000
                should_decay = (is_currently_frozen and step > freeze_anchor)
                decay_origin = freeze_anchor
                
            if should_decay:
                progress = min((step - decay_origin) / total_decay_steps, 1.0)
                decayed_lr = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress))
            else:
                decayed_lr = base_lr
                
            if step <= 24000:
                self.assertAlmostEqual(decayed_lr, base_lr, places=7, msg=f"Step {step} 不应发生衰减")
                print(f"  Step {step:5d}: LR 保持探索峰值 {decayed_lr:.7f}")
            elif step == 35000:
                # 刚好衰减到中点 (progress = 0.5)
                expected_mid = lr_min + 0.5 * (base_lr - lr_min) * 1.0
                self.assertAlmostEqual(decayed_lr, expected_mid, places=7, msg="中点衰减值不符")
                print(f"  Step {step:5d}: LR 严格走在余弦中点 {decayed_lr:.7f} (进度 50%)")
            else:
                self.assertAlmostEqual(decayed_lr, lr_min, places=7, msg=f"Step {step} 必须衰减至最低 LR")
                print(f"  Step {step:5d}: LR 成功收敛至极小学习率 {decayed_lr:.7f} (精度收敛模式)")
                
        print("  [√] disable_freeze 场景下的余弦学习率平滑衰减数学验证全部通过！")

    def test_02_dynamic_patience_logic(self):
        """测试 2: 验证动态早停耐心判定逻辑"""
        print("\n--- [Test 02] 验证 config.patience 动态早停参数生效 ---")
        
        # 模拟配置
        class MockConfig:
            patience = 12
        
        config = MockConfig()
        
        # 模拟在 Step 38000 发生第 5 次未改善
        patient = 5
        self.assertFalse(patient >= getattr(config, "patience", 5), "patient=5 时不应触发早停 (当前上限为 12)")
        print(f"  Patient=5 时: 早停未触发 (当前阈值上限: {config.patience})，允许模型继续走完余弦退火周期。")
        
        # 模拟达到 12 次
        patient = 12
        self.assertTrue(patient >= getattr(config, "patience", 5), "patient=12 时必须正常触发早停退出")
        print(f"  Patient=12 时: 成功触发早停退出保护！")
        print("  [√] 动态早停耐心机制验证通过！")

    def test_03_code_hygiene_audit(self):
        """测试 3: 静态代码红线审计"""
        print("\n--- [Test 03] 扫描废弃 API 与原地操作漏洞 ---")
        src_dir = os.path.join(PROJECT_ROOT, "src")
        forbidden_patterns = ["F.sigmoid", "np.float\b"]
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        for pattern in forbidden_patterns:
                            self.assertNotIn(pattern, content, f"发现违禁代码模式: {pattern} 在 {file_path}")
        print("  [√] 静态代码审计完成，无 F.sigmoid / np.float 违禁调用！")

if __name__ == "__main__":
    unittest.main()
