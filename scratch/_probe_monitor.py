# -*- coding: utf-8 -*-
"""T11-B 监控代码快速结构探针（不训练）"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ["probe", "--dataset", "ED", "--max_grad_norm", "6.0"]
import torch
from src.utils import config as config_mod
config = config_mod.config
print("[probe] device =", config.device, "| dataset =", config.dataset,
      "| max_grad_norm =", config.max_grad_norm,
      "| use_pcam =", getattr(config, "use_pcam", None),
      "| hop =", getattr(config, "hop", None))

from src.models.CASE.model import CASE

class Holder(torch.nn.Module):
    pass

h = Holder()
h._gn_buf = torch.zeros(20000, device=config.device)
h._gn_n = 0
h._pcg_conflict_steps = 0
h._pcg_cos_sum = 0.0

print("\n[A] 监控属性就绪:", tuple(h._gn_buf.shape), h._gn_buf.device, "| _gn_n =", h._gn_n)
print("[A] '_gn_buf' in state_dict ? ->", "_gn_buf" in h.state_dict(),
      "(必须 False)")

vals = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.], device=config.device)
h._gn_buf[:10] = vals
h._gn_n = 10
h._pcg_conflict_steps = 3
h._pcg_cos_sum = 0.45
print("\n[B] 手工填入范数 [1..10]；期望 mean=5.5 | 中位数=5.5 | P95=10 | max=10 | >2.0=80.00% | >6.0=40.00%")
print("[B] 期望 PCGrad：冲突步=3/10 (30.00%) | 平均|cos|=0.1500")
CASE.print_training_monitor(h)
print("[probe] 完成")
