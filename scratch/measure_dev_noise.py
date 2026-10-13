# -*- coding: utf-8 -*-
"""
CASE-EPCL V9 阶段 0 追加实验：验证集批次重组噪声宽度实测 (measure_dev_noise.py)

【实验目的】
验证集 DataLoader 在 loader.py 中被配置为 shuffle=True + batch_size=config.batch_size，
而 common.evaluate 对 PPL 与 EMO_loss 均采用"批次级宏平均"聚合，导致同一份权重在不同
随机批次划分下会得到不同的验证指标。本脚本通过多种子重复测量，量化该噪声的分布宽度，
并与 Trial 9 的检查点遴选裕度（PPL 0.0517 / 复合评分 0.1258）直接对比。

【对照组设计】
- 实验组：dev 集（shuffle=True, batch=32）→ 预期出现实现间波动
- 阴性对照：test 集（shuffle=False, batch=1）→ 预期波动为 0，用于证明测量方法本身
  不存在额外随机源（若 test 也出现波动，则说明模型前向含有未知随机性，须另行排查）

【口径对照】
- macro：exp(mean_b(ctx_loss_b))             现行实现口径（批次等权，受划分影响）
- micro：exp(Σ_b ctx_loss_b · n_b / Σ_b n_b)  token 级加权（与批次划分无关）

【判读原则】
两次独立实现的差值标准差 ≈ sigma * sqrt(2)，因此不要把单次实现的 sigma 直接与
遴选裕度相减比较，应比较"差值分布"与裕度。

【用法】
python scratch/measure_dev_noise.py \
  --model_path save/v9_trial9/CASE_7499_37.7145_composite_best \
  --dev_repeats 10 --test_repeats 2
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import sys
import math
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# 项目路径注入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 复用已封版诊断脚本的拓扑自适应加载逻辑（diag_pcam_abc 模块级无副作用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_pcam_abc import safe_load_checkpoint  # noqa: E402
from src.utils.config import config  # noqa: E402

# Trial 9 的遴选裕度（用于判读，数据来源：logs/v9_trial9.log）
MARGIN_PPL = 0.0517      # Step 6999 (37.6628) 与 Step 7499 (37.7145) 的 PPL 差
MARGIN_SCORE = 0.1258    # 49.0543 与 48.9285 的复合评分差


def collect_metrics(model, dataloader, desc):
    """
    单轮全量前向采集：返回批次级 ctx_loss、emotion_loss 与非 PAD token 数
    ctx_loss 为 batch 内非 PAD token 的平均 NLL（见 model.py 的 criterion_ppl 定义）
    """
    ctx_list, emo_list, tok_list = [], [], []
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
            try:
                (_, _, _, ctx_loss, _, _, _, emo_loss, _, _, _) = model.train_one_batch(
                    batch, 0, train=False
                )
            except RuntimeError as e:
                err_msg = str(e).lower()
                if ("out of memory" in err_msg or "cuda error" in err_msg) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        (_, _, _, ctx_loss, _, _, _, emo_loss, _, _, _) = model.train_one_batch(
                            batch, 0, train=False
                        )
                    except RuntimeError as e2:
                        torch.cuda.empty_cache()
                        # 单批次 CPU Fallback 防御 (保证 4GB 显卡 100% 顺畅处理极端常识图 Batch)
                        orig_device = config.device
                        config.device = "cpu"
                        model_cpu = model.to("cpu")
                        batch_cpu = {
                            k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()
                        }
                        (_, _, _, ctx_loss, _, _, _, emo_loss, _, _, _) = model_cpu.train_one_batch(
                            batch_cpu, 0, train=False
                        )
                        config.device = orig_device
                        model.to(orig_device)
                        torch.cuda.empty_cache()
                else:
                    raise e
            # 与 model.py 中 dec_batch.ne(config.PAD_idx) 同源，用于 token 级加权
            n_tok = int((batch["target_batch"] != config.PAD_idx).sum().item())
            ctx_list.append(float(ctx_loss))
            emo_list.append(float(emo_loss))
            tok_list.append(n_tok)
            if (batch_idx + 1) % 15 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.array(ctx_list), np.array(emo_list), np.array(tok_list)


def summarize(ctx, emo, tok, emo_loss_weight=5.0):
    """
    计算 macro / micro 两种口径的 PPL，以及复合评分（与 main.py 的 emo_loss 模式一致）
    """
    macro_ppl = math.exp(float(ctx.mean()))
    micro_ppl = math.exp(float((ctx * tok).sum() / max(float(tok.sum()), 1.0)))
    emo_loss = float(emo.mean())
    return {
        "macro_ppl": macro_ppl,
        "micro_ppl": micro_ppl,
        "emo_loss": emo_loss,
        "score": macro_ppl + emo_loss_weight * emo_loss,
    }


def run_repeats(model, dataloader, repeats, base_seed, desc):
    """多种子重复测量；每次重复前重置全局 RNG，使 shuffle 排列发生改变"""
    rows = []
    for i in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.manual_seed(base_seed + i)
        np.random.seed(base_seed + i)
        ctx, emo, tok = collect_metrics(model, dataloader, desc=f"{desc} #{i + 1}")
        row = summarize(ctx, emo, tok)
        row["seed"] = base_seed + i
        row["batches"] = int(len(ctx))
        row["tokens"] = int(tok.sum())
        rows.append(row)
        print(
            f"    [{desc} #{i + 1}] seed={row['seed']} | macro_PPL={row['macro_ppl']:.4f} | "
            f"micro_PPL={row['micro_ppl']:.4f} | EMO_loss={row['emo_loss']:.4f} | "
            f"Score={row['score']:.4f}"
        )
    return rows


def stats(rows, key):
    """样本统计量；n=1 时标准差记 0"""
    vals = np.array([r[key] for r in rows], dtype=float)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "range": float(vals.max() - vals.min()),
    }


def run(args):
    print("=" * 78)
    print(" CASE-EPCL V9 阶段 0 追加实验：验证集批次重组噪声宽度实测")
    print("=" * 78)

    if args.device:
        config.device = args.device
        print(f"[*] 显式指定运行设备: {args.device}")

    model, vocab, dev_set, test_set = safe_load_checkpoint(args.model_path, batch_size=args.batch_size)
    if args.device:
        model = model.to(args.device)
    print(f"[*] dev 批次数 = {len(dev_set)} | test 批次数 = {len(test_set)}")

    print("\n[*] 实验组：dev 集重复测量")
    dev_rows = run_repeats(model, dev_set, args.dev_repeats, config.seed, "Dev")

    print("\n[*] 阴性对照：test 集（shuffle=False）重复测量")
    test_rows = run_repeats(model, test_set, args.test_repeats, config.seed, "Test")

    dev_stat = {k: stats(dev_rows, k) for k in ["macro_ppl", "micro_ppl", "emo_loss", "score"]}
    test_stat = {k: stats(test_rows, k) for k in ["macro_ppl", "micro_ppl", "emo_loss", "score"]}

    # 判读：把 dev 的实现间波动与 Trial 9 的遴选裕度并列
    macro_range = dev_stat["macro_ppl"]["range"]
    macro_std = dev_stat["macro_ppl"]["std"]
    micro_std = dev_stat["micro_ppl"]["std"]
    score_range = dev_stat["score"]["range"]
    test_std = test_stat["macro_ppl"]["std"]

    verdict = {
        "test_deterministic": bool(test_std < 1e-6),
        "batch_grouping_noise_confirmed": bool(macro_range > MARGIN_PPL or macro_std > 0.02),
        "macro_noise_exceeds_margin": bool(macro_range > MARGIN_PPL),
        "score_noise_exceeds_margin": bool(score_range > MARGIN_SCORE),
        "micro_is_partition_invariant": bool(micro_std < 1e-6),
        "margin_ppl": MARGIN_PPL,
        "margin_score": MARGIN_SCORE,
    }

    report = {
        "checkpoint": args.model_path,
        "dev_repeats": args.dev_repeats,
        "test_repeats": args.test_repeats,
        "dev_rows": dev_rows,
        "test_rows": test_rows,
        "dev_stats": dev_stat,
        "test_stats": test_stat,
        "verdict": verdict,
    }

    output_dir = os.path.join(PROJECT_ROOT, "save", "diag")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "dev_noise_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("                     批次重组噪声测量结论")
    print("=" * 78)
    print(f" 1. dev macro_PPL : 均值={dev_stat['macro_ppl']['mean']:.4f} | "
          f"标准差={macro_std:.4f} | 极差={macro_range:.4f} | "
          f"区间=[{dev_stat['macro_ppl']['min']:.4f}, {dev_stat['macro_ppl']['max']:.4f}]")
    print(f" 2. dev micro_PPL : 均值={dev_stat['micro_ppl']['mean']:.4f} | "
          f"标准差={micro_std:.4f} | 极差={dev_stat['micro_ppl']['range']:.4f}")
    print(f" 3. dev Score     : 标准差={dev_stat['score']['std']:.4f} | 极差={score_range:.4f}")
    print(f" 4. test 阴性对照 : macro_PPL 标准差={test_std:.6f} | 极差={test_stat['macro_ppl']['range']:.6f}")
    print(f" 5. Trial 9 遴选裕度: PPL={MARGIN_PPL:.4f} | Score={MARGIN_SCORE:.4f}")
    print("-" * 78)

    if not verdict["test_deterministic"]:
        print(" [!] test 阴性对照未通过：test 路径本应严格确定却出现波动，须先排查模型前向的随机源。")
    if verdict["macro_noise_exceeds_margin"]:
        print(" [结论 1] dev 批次重组噪声极差 > 遴选裕度：Trial 9 的 composite_best 遴选落在噪声带内。")
    else:
        print(" [结论 1] dev 批次重组噪声未超过 PPL 遴选裕度：遴选结果对该噪声相对稳健。")
    if verdict["score_noise_exceeds_margin"]:
        print(" [结论 2] 复合评分的实现间极差 > 遴选裕度：检查点优劣判断受批次划分主导。")
    else:
        print(" [结论 2] 复合评分极差未超过遴选裕度。")
    if verdict["micro_is_partition_invariant"]:
        print(" [结论 3] micro（token 级加权）口径与批次划分无关，可作为验证指标的确定性替代口径。")
    else:
        print(" [!] micro 口径仍出现波动，说明 ctx_loss 并非严格的 token 级均值，需复核 criterion_ppl。")

    print("-" * 78)
    print(" [NOISE_COMPACT_JSON]")
    print(json.dumps({"dev_stats": dev_stat, "test_stats": test_stat, "verdict": verdict},
                     ensure_ascii=False))
    print(f"\n[*] 完整结果已落盘: {report_file}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证集批次重组噪声宽度实测")
    parser.add_argument("--model_path", type=str,
                        default="save/v9_trial9/CASE_7499_37.7145_composite_best",
                        help="目标检查点路径")
    parser.add_argument("--batch_size", type=int, default=32, help="前向评估批次大小")
    parser.add_argument("--device", type=str, default=None, help="指定计算设备 (如 cpu 或 cuda)")
    parser.add_argument("--dev_repeats", type=int, default=10, help="dev 集重复测量次数")
    parser.add_argument("--test_repeats", type=int, default=2, help="test 集重复测量次数（阴性对照）")
    args = parser.parse_args()
    run(args)
