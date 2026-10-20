"""T12 云端/本地先验预检（只读，不训练、不落盘到正式目录）。

用途：核对 Gate 3 三项 —— 同一 prior 的 SHA256、sklearn 版本、KMeans 实际簇成员。
与 docs/v9/reports/v9_trial12_local_validation.md:63 的预检等价，但显式注入 argparse 参数，
避免 `python -c` 位置参数不生效的问题。

运行：
  python scratch/_t12_preflight.py <prior_ckpt> [P]
"""
import os
import sys
import hashlib

import numpy as np
import sklearn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    prior = sys.argv[1] if len(sys.argv) > 1 else ""
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    assert prior and os.path.exists(prior), f"prior 不存在: {prior}"

    # 显式注入 argparse 参数（关键：不依赖 `-c` 后的位置参数）
    sys.argv = [
        "preflight",
        "--proto_prefix_prior_path", prior,
        "--proto_prefix_len", str(P),
        "--proto_prefix_init", "proto",
        "--proto_prefix_source", "ckpt",
        "--use_proto_prefix",
    ]
    from src.utils.config import config
    from src.models.CASE.model import CASE

    print(f"[preflight] prior      = {prior}")
    print(f"[preflight] size       = {os.path.getsize(prior)} bytes")
    print(f"[preflight] sha256     = {sha256_of(prior)}")
    print(f"[preflight] sklearn    = {sklearn.__version__}")
    print(f"[preflight] numpy      = {np.__version__}")
    print(f"[preflight] P          = {P}, hidden demanded = {config.hidden_dim}")

    proto = CASE._load_proto_from_ckpt(None)           # (32, 300)
    centers = CASE._deterministic_project(None, proto, P)
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()

    print(f"[preflight] proto      shape={tuple(proto.shape)}")
    print(f"[preflight] centers    shape={tuple(centers.shape)}")
    print(f"[preflight] centers    L2 = {[round(float(np.linalg.norm(c)), 4) for c in centers]}")

    # 簇成员：用同参数再聚一次，取 labels（确定性：random_state=13, n_init=20）
    from sklearn.cluster import KMeans
    pn = proto / proto.norm(dim=1, keepdim=True).clamp_min(1e-8)
    km = KMeans(n_clusters=P, random_state=13, n_init=20)
    labels = km.fit_predict(pn.numpy())
    print("[preflight] cluster_members (label: [class indices])")
    for c in range(P):
        members = [int(i) for i in np.where(labels == c)[0]]
        print(f"  cluster {c}: n={len(members)} members={members}")
    print("[preflight] sizes =", [int((labels == c).sum()) for c in range(P)])
    print("[preflight] DONE (read-only, 未落盘正式 npz)")


if __name__ == "__main__":
    main()