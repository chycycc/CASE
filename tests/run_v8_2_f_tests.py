# -*- coding: utf-8 -*-
"""V8.2 F 测试运行器 (无需 pytest)"""
import sys
import os
import math
import unittest

# 注册工程根路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 直接导入测试模块
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))
from test_v8_2_f_integration import (
    TestBellShapedRamp,
    TestBiasAnnealingMinScale,
    TestParetoCompositeScore,
    TestProtectionPeriod,
    TestBellShapeModelIntegration
)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestBellShapedRamp, TestBiasAnnealingMinScale,
                TestParetoCompositeScore, TestProtectionPeriod,
                TestBellShapeModelIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
