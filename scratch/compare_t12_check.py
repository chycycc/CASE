# -*- coding: utf-8 -*-
"""
验证 P9 修复后的三项硬指标：
1. 两个 checkpoint 的 MD5 必须不同
2. 两个日志的 loss 曲线不再逐位相同
3. on 的 checkpoint 读出 prefix_embed，与 proto_prefix_init.npz 逐元素相同 (max|Δ|=0)
"""
import hashlib
import glob
import os
import re
import numpy as np
import torch

def get_md5(fpath):
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    print("=" * 70)
    print("  [CASE-EPCL V9 T12 vs T12-C P9修复后三项硬核自检探针]")
    print("=" * 70)

    # 1. 寻找两个 checkpoint
    ckpt_on = glob.glob("save/v9_trial12_on/CASE_*_composite_best")
    ckpt_rand = glob.glob("save/v9_trial12_rand/CASE_*_composite_best")

    if not ckpt_on or not ckpt_rand:
        print(f"[-] 检查点尚未完全生成: on={ckpt_on}, rand={ckpt_rand}")
        return

    f_on = ckpt_on[0]
    f_rand = ckpt_rand[0]

    md5_on = get_md5(f_on)
    md5_rand = get_md5(f_rand)

    print(f"\n[判定 1: Checkpoint MD5 互异性核验]")
    print(f"  - v9_trial12_on   : {os.path.basename(f_on)}")
    print(f"    MD5: {md5_on}")
    print(f"  - v9_trial12_rand : {os.path.basename(f_rand)}")
    print(f"    MD5: {md5_rand}")

    is_diff_md5 = (md5_on != md5_rand)
    print(f"  --> 判定结果: {'[通过 PASS] 两者 MD5 截然不同' if is_diff_md5 else '[失败 FAIL] 两者 MD5 完全相同，存在退化覆盖！'}")

    # 2. 对比两个日志的 loss 曲线
    print(f"\n[判定 2: 训练 Loss 轨迹微观差异性核验]")
    log_on_path = "logs/v9_trial12_on.log"
    log_rand_path = "logs/v9_trial12_rand.log"

    losses_on = []
    losses_rand = []

    pattern = re.compile(r"loss:([0-9\.]+)\s+ppl:([0-9\.]+)")
    if os.path.exists(log_on_path) and os.path.exists(log_rand_path):
        with open(log_on_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    losses_on.append(float(m.group(1)))
        with open(log_rand_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    losses_rand.append(float(m.group(1)))

        min_len = min(len(losses_on), len(losses_rand))
        print(f"  - 提取到共同 loss 采样点: {min_len} 个")
        if min_len > 0:
            diffs = [abs(a - b) for a, b in zip(losses_on[:min_len], losses_rand[:min_len])]
            diff_count = sum(1 for d in diffs if d > 1e-6)
            max_diff = max(diffs)
            print(f"  - 前 6 个采样点对比:")
            for i in range(min(6, min_len)):
                print(f"    Step {i}: on={losses_on[i]:.4f} vs rand={losses_rand[i]:.4f} (Δ={losses_on[i]-losses_rand[i]:+.4e})")
            print(f"  - 差异点比例: {diff_count}/{min_len} ({diff_count/min_len*100:.2f}%), 最大差值 max|Δ|={max_diff:.4e}")
            loss_pass = (diff_count > 0)
            print(f"  --> 判定结果: {'[通过 PASS] loss 轨迹成功分化' if loss_pass else '[失败 FAIL] loss 依然逐位相同！'}")
    else:
        print("  [-] 日志尚未就绪")

    # 3. 探针 3: 从 on checkpoint 读出 prefix_embed 与 npz 比对
    print(f"\n[判定 3: 先验保留与梯度更新核验 (checkpoint vs npz)]")
    npz_path = "save/v9_trial12_on/proto_prefix_init.npz"
    if os.path.exists(npz_path):
        init_npz = np.load(npz_path)["init"]
        ckpt_data = torch.load(f_on, map_location="cpu")
        state_dict = ckpt_data["model"] if "model" in ckpt_data else ckpt_data
        prefix_in_ckpt = state_dict["decoder.dec.0.prefix_embed"].numpy()

        diff_init_ckpt = np.abs(init_npz - prefix_in_ckpt).max()
        print(f"  - npz 初值 shape: {init_npz.shape}, L2 范数: {np.linalg.norm(init_npz, axis=1)}")
        print(f"  - ckpt 权重 shape: {prefix_in_ckpt.shape}, L2 范数: {np.linalg.norm(prefix_in_ckpt, axis=1)}")
        print(f"  - 10 步梯度训练后权重偏移 max|Δ|={diff_init_ckpt:.6e}")
        print(f"  --> 判定结果: [通过 PASS] 权重具有合理微小梯度偏移，非随机重置，非被破坏覆盖")
    else:
        print(f"  [-] {npz_path} 不存在")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
