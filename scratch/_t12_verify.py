# -*- coding: utf-8 -*-
"""[T12 独立复核] 只读验证：先验原型 Gram / KMeans 簇成员 / 中心正交性 / 确定性。不写任何项目文件。"""
import os
import sys
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ED = ["surprised", "excited", "annoyed", "proud", "angry", "sad", "grateful", "lonely",
      "impressed", "afraid", "disgusted", "confident", "terrified", "hopeful", "anxious",
      "disappointed", "joyful", "prepared", "guilty", "furious", "nostalgic", "jealous",
      "anticipating", "embarrassed", "content", "devastated", "sentimental", "caring",
      "trusting", "ashamed", "apprehensive", "faithful"]


def load_proto(path):
    state = torch.load(path, map_location="cpu")
    sd = state["model"] if "model" in state else state
    key = next((k for k in sd if k.endswith("epcl_criterion.prototypes")), None)
    if key is None:
        return None, None
    return key, sd[key].detach().float().clone()


def report(tag, path):
    print("=" * 78)
    print(f"[{tag}] {path}  exists={os.path.exists(path)}")
    if not os.path.exists(path):
        print("  SKIP (file missing)")
        return
    key, proto = load_proto(path)
    print(f"  key={key}  shape={tuple(proto.shape)}")
    gram = proto @ proto.t()
    off = gram[~torch.eye(32, dtype=torch.bool)]
    print(f"  Gram off-diag: mean={off.mean():.6f} median={off.median():.6f} "
          f"min={off.min():.4f} max={off.max():.4f}")

    km = KMeans(n_clusters=4, random_state=13, n_init=20).fit(proto.numpy())
    centers = F.normalize(torch.from_numpy(km.cluster_centers_).float(), p=2, dim=1)
    labels = km.labels_
    for c in range(4):
        mem = [ED[i] for i in range(32) if labels[i] == c]
        print(f"  cluster {c}: n={len(mem):2d} {mem}")
    cc = (centers @ centers.t()).numpy()
    print("  center pairwise cos (rounded):")
    for r in range(4):
        print("    " + " ".join(f"{cc[r][k]:+.3f}" for k in range(4)))
    print(f"  center L2 norms: {[round(v, 4) for v in centers.norm(dim=1).tolist()]}")

    km2 = KMeans(n_clusters=4, random_state=13, n_init=20).fit(proto.numpy())
    c2 = F.normalize(torch.from_numpy(km2.cluster_centers_).float(), p=2, dim=1)
    print(f"  determinism: max|dE|={ (c2 - centers).abs().max().item():.3e}  "
          f"labels_equal={(km2.labels_ == labels).all()}")


if __name__ == "__main__":
    report("T10", "results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best")
    report("T11-B", "results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best")
