import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
from src.utils.constants import ED_MAP_EMO

ckpt_path = 'results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best'
ckpt = torch.load(ckpt_path, map_location='cpu')
state = ckpt.get('model', ckpt)
protos = state['epcl_criterion.prototypes'] # [32, d]
print(f"Loaded prototypes from T10: shape={protos.shape}, norm={protos.norm(dim=-1).mean().item():.4f}")

# 归一化后计算真实的 P * P^T
P_norm = F.normalize(protos, p=2, dim=-1)
sim_matrix = torch.matmul(P_norm, P_norm.T).numpy()

# 提取非对角线元素
n_classes = sim_matrix.shape[0]
off_diag = []
pairs = []
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        off_diag.append(sim_matrix[i, j])
        pairs.append((sim_matrix[i, j], ED_MAP_EMO.get(i, str(i)), ED_MAP_EMO.get(j, str(j))))

off_diag = np.array(off_diag)
print("\n=== T10 真实原型矩阵 P·P^T (直接计算，非代理) ===")
print(f"非对角线余弦均值: {np.mean(off_diag):.4f}")
print(f"非对角线余弦中位数: {np.median(off_diag):.4f}")
print(f"非对角线余弦标准差: {np.std(off_diag):.4f}")
print(f"非对角线最小/最大值: {np.min(off_diag):.4f} / {np.max(off_diag):.4f}")
print(f"余弦 > 0.5 的近邻原型对占比: {(off_diag > 0.5).mean() * 100:.2f}% (共 {sum(off_diag > 0.5)}/{len(off_diag)} 对)")

pairs.sort(key=lambda x: x[0], reverse=True)
print("\n--- 真实原型间余弦相似度最高 Top 10 (最重叠对) ---")
for sim, c1, c2 in pairs[:10]:
    print(f"  {c1:<15} ~ {c2:<15}: cos = {sim:.4f}")

print("\n--- 真实原型间余弦相似度最低 Top 5 (最正交/负相关对) ---")
for sim, c1, c2 in pairs[-5:]:
    print(f"  {c1:<15} ~ {c2:<15}: cos = {sim:.4f}")
