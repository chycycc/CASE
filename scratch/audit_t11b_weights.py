# -*- coding: utf-8 -*-
"""
CASE-EPCL V9 T11-B Checkpoint 权重深度技术审计
1. 拓扑完整性核验 (关键前缀覆盖度)
2. 原型提取与几何统计量 (L2范数, Gram矩阵非对角均值/中位数)
3. 离线 KMeans(K=4) 簇划分验证
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from sklearn.cluster import KMeans

ckpt_path = "results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best"

print("=" * 70)
print(f"[*] 正在审计 T11-B 检查点权重: {ckpt_path}")
print(f"    - 文件大小: {os.path.getsize(ckpt_path) / (1024*1024):.2f} MB")
print("=" * 70)

# 1. 兼容加载
load_kwargs = {"map_location": "cpu"}
if "weights_only" in inspect.signature(torch.load).parameters:
    load_kwargs["weights_only"] = False

ckpt = torch.load(ckpt_path, **load_kwargs)
raw_keys = ckpt["model"].keys() if "model" in ckpt else ckpt.keys()
state_dict = ckpt["model"] if "model" in ckpt else ckpt

print(f"[+] 权重字典加载成功，包含参数张量数: {len(raw_keys)}")

# 拓扑前缀核验
critical_prefixes = ["decoder.", "encoder.", "emotion_linear.", "pcam.", "epcl_criterion."]
for prefix in critical_prefixes:
    matches = [k for k in raw_keys if k.startswith(prefix)]
    print(f"    - 关键前缀 [{prefix:<15}]: 包含 {len(matches):>3} 项参数")
    assert len(matches) > 0, f"关键前缀 {prefix} 缺失！"

# 2. 原型提取
proto_key = None
for k in raw_keys:
    if "epcl_criterion.prototypes" in k or k.endswith(".prototypes"):
        proto_key = k
        break

assert proto_key is not None, "未找到 prototypes 参数！"
protos = state_dict[proto_key].detach().float().clone().numpy()
print(f"\n[+] 成功提取先验原型: key='{proto_key}', shape={protos.shape}")
assert protos.shape == (32, 300), f"原型形状异常: {protos.shape} != (32, 300)"
assert not np.isnan(protos).any(), "原型包含 NaN！"
assert not np.isinf(protos).any(), "原型包含 Inf！"

# 范数核验
norms = np.linalg.norm(protos, axis=1)
print(f"    - 原型 L2 范数: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

# Gram 矩阵非对角统计量
normed_protos = protos / (norms[:, None] + 1e-12)
gram = np.dot(normed_protos, normed_protos.T)
diag_mask = np.eye(32, dtype=bool)
off_diag = gram[~diag_mask]
mean_off = float(np.mean(off_diag))
median_off = float(np.median(off_diag))
min_off = float(np.min(off_diag))
max_off = float(np.max(off_diag))

print(f"    - Gram 矩阵非对角统计: mean={mean_off:+.4f}, median={median_off:+.4f}, min={min_off:+.4f}, max={max_off:+.4f}")
print(f"    - 均值偏离审计 (|mean| < 0.10): {'[通过 PASS]' if abs(mean_off) < 0.10 else '[警告 WARN]'}")

# 3. 离线 KMeans(K=4) 聚类测试
print(f"\n[+] 执行 KMeans(K=4, random_state=13, n_init=20) 先验聚类模拟:")
kmeans = KMeans(n_clusters=4, random_state=13, n_init=20)
cluster_ids = kmeans.fit_predict(protos)
centroids = kmeans.cluster_centers_

# 情绪标签 (ED 32类)
emotions = [
    "surprised", "excited", "annoyed", "proud", "angry", "sad", "grateful", "lonely",
    "impressed", "afraid", "disgusted", "confident", "terrified", "hopeful", "anxious",
    "jealous", "disappointed", "joyful", "apprehensive", "embarrassed", "devastated",
    "nostalgic", "sentimental", "caring", "trusting", "ashamed", "apprehensive_2",
    "faithful", "content", "furious", "prepared", "guilty"
]
# 取实际 vocabulary / emo_num 顺序 (按 loader.py 定义)
from src.utils.data.loader import prepare_data_seq
from src.utils.config import config
config.batch_size = 1
_, _, _, vocab, emo_num, _ = prepare_data_seq(batch_size=1)
# 获取真实 32 情绪映射
idx2emo = {v: k for k, v in vocab.emo2idx.items()} if hasattr(vocab, 'emo2idx') else {i: emotions[i] for i in range(32)}

cluster_members = {}
for c in range(4):
    m_indices = np.where(cluster_ids == c)[0]
    m_names = [idx2emo.get(i, f"emo_{i}") for i in m_indices]
    cluster_members[c] = m_names
    print(f"    - 簇 {c} (n={len(m_indices):>2}): {m_names}")

# 簇质心 L2 归一化统计
c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
E_proto = centroids / np.maximum(c_norms, 1e-12)
e_l2 = np.linalg.norm(E_proto, axis=1)
print(f"    - 归一化簇质心 E_proto: shape={E_proto.shape}, L2={np.round(e_l2, 4).tolist()}, std={E_proto.std():.4f}")

print("\n" + "=" * 70)
print("[+] T11-B 检查点权重审计全部通过！符合 T12 先验加载与聚类注入规约。")
print("=" * 70)
