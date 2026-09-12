# -*- coding: utf-8 -*-
"""
[CASE-EPCL 路线 A 核心评测脚本]
多样性自适应解码评测 (Diverse Adaptive Decoding Evaluation)
- 覆盖策略:
  1. Greedy Search (确定性基准)
  2. Standard Beam Search (k=5, 原始尖锐分布易重复)
  3. Beam Search + No-Repeat 3-gram (k=5, 局部短语硬阻塞)
  4. Nucleus Sampling (T=0.5, top_p=0.9, 保守采样)
  5. Nucleus Sampling (T=0.7, top_p=0.9, 温和推荐)
  6. Nucleus Sampling (T=1.0, top_p=0.9, 标准采样)
- 评测指标:
  Distinct-1 (%), Distinct-2 (%), Unique Sentences (%), Avg Length, Case Comparison
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import math
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nltk import word_tokenize

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq
from src.utils.decode.case import Translator


def calc_distinct_n(n, candidates):
    """按照 CASE 原版评测口径计算 Distinct-n (%)"""
    d = {}
    total = 0
    tokenized = [word_tokenize(c) for c in candidates]
    for s in tokenized:
        for i in range(len(s) - n + 1):
            key = tuple(s[i : i + n])
            d[key] = 1
            total += 1
    return (len(d) / (total + 1e-16)) * 100.0


def calc_metrics(candidates):
    """计算文本集合的多样性与长度指标"""
    if not candidates:
        return {"Dist-1": 0.0, "Dist-2": 0.0, "Unique": 0.0, "Avg_Len": 0.0}
    
    dist1 = calc_distinct_n(1, candidates)
    dist2 = calc_distinct_n(2, candidates)
    unique_rate = (len(set(candidates)) / (len(candidates) + 1e-16)) * 100.0
    
    tokenized = [word_tokenize(c) for c in candidates]
    avg_len = np.mean([len(s) for s in tokenized]) if tokenized else 0.0
    
    return {
        "Dist-1": round(dist1, 2),
        "Dist-2": round(dist2, 2),
        "Unique": round(unique_rate, 2),
        "Avg_Len": round(avg_len, 2),
    }


def load_eval_model(checkpoint_path):
    """加载模型结构与最佳 checkpoint 权重"""
    print(f"[*] 正在配置评测环境与模型结构参数...")
    config.dataset = "ED"
    config.woStrategy = True
    config.num_prototypes_per_class = 2
    config.emotion_head_type = "residual_mlp"
    config.use_pcam = True
    config.pcam_heads = 2
    config.pcam_dropout = 0.1
    config.pcam_gate_bias = -1.0
    config.test = True
    config.model = "case"
    config.beam_size = 5
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed()

    print("[*] 正在加载数据与词表 (prepare_data_seq)...")
    _, _, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=1)

    print(f"[*] 实例化 CASE 模型 (is_eval=True)...")
    model = CASE(
        vocab,
        emotion_num=emo_num,
        strategy_num=strategy_num,
        is_eval=True,
        model_file_path=None,
    )
    model.to(config.device)

    print(f"[*] 正在从 {checkpoint_path} 加载黄金权重...")
    state = torch.load(checkpoint_path, map_location=config.device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    model.is_eval = True

    translator = Translator(model, vocab)
    return model, vocab, test_set, translator


def run_evaluation(model, vocab, test_set, translator, num_samples=500, max_dec_step=50):
    """在测试集上对比执行多种解码策略并统计指标"""
    total_in_set = len(test_set)
    limit = total_in_set if num_samples <= 0 else min(num_samples, total_in_set)
    print(f"\n=======================================================")
    print(f"  路线 A: 多样性自适应解码评测启动 (样本数: {limit} / {total_in_set})")
    print(f"=======================================================\n")

    strategies = {
        "Greedy": [],
        "Beam_k5 (Standard)": [],
        "Beam_k5_no_repeat3": [],
        "Sampling_T0.5_p0.9": [],
        "Sampling_T0.7_p0.9": [],
        "Sampling_T1.0_p0.9": [],
    }

    contexts = []
    references = []
    emotions = []

    pbar = tqdm(total=limit, desc="Decoding Samples", dynamic_ncols=True)

    with torch.no_grad():
        for i, batch in enumerate(test_set):
            if i >= limit:
                break

            rf = " ".join(batch["target_txt"][0])
            ctx = [" ".join(s) for s in batch["input_txt"][0]]
            emo = batch["program_txt"][0]

            references.append(rf)
            contexts.append(ctx)
            emotions.append(emo)

            # 1. Greedy 解码
            sent_greedy = model.decoder_greedy(batch, max_dec_step=max_dec_step)[0]
            strategies["Greedy"].append(sent_greedy)

            # 2. Standard Beam Search (k=5, n=0)
            sent_beam_std = translator.beam_search(batch, max_dec_step=max_dec_step, no_repeat_ngram_size=0)[0]
            strategies["Beam_k5 (Standard)"].append(sent_beam_std)

            # 3. Beam Search + no-repeat-3-gram (k=5, n=3)
            sent_beam_nr3 = translator.beam_search(batch, max_dec_step=max_dec_step, no_repeat_ngram_size=3)[0]
            strategies["Beam_k5_no_repeat3"].append(sent_beam_nr3)

            # 4. Nucleus Sampling (T=0.5, p=0.9)
            sent_samp_t05 = model.decoder_sampling(batch, max_dec_step=max_dec_step, temp=0.5, top_p=0.9)[0]
            strategies["Sampling_T0.5_p0.9"].append(sent_samp_t05)

            # 5. Nucleus Sampling (T=0.7, p=0.9)
            sent_samp_t07 = model.decoder_sampling(batch, max_dec_step=max_dec_step, temp=0.7, top_p=0.9)[0]
            strategies["Sampling_T0.7_p0.9"].append(sent_samp_t07)

            # 6. Nucleus Sampling (T=1.0, p=0.9)
            sent_samp_t10 = model.decoder_sampling(batch, max_dec_step=max_dec_step, temp=1.0, top_p=0.9)[0]
            strategies["Sampling_T1.0_p0.9"].append(sent_samp_t10)

            pbar.update(1)

    pbar.close()

    # 指标汇总计算
    results_summary = {}
    print("\n" + "=" * 80)
    print(f"{'解码策略':<25} | {'Dist-1 (%)':<10} | {'Dist-2 (%)':<10} | {'Unique (%)':<10} | {'Avg Len':<8}")
    print("-" * 80)

    for strat_name, sents in strategies.items():
        metrics = calc_metrics(sents)
        results_summary[strat_name] = metrics
        print(f"{strat_name:<25} | {metrics['Dist-1']:<10.2f} | {metrics['Dist-2']:<10.2f} | {metrics['Unique']:<10.2f} | {metrics['Avg_Len']:<8.2f}")
    print("=" * 80 + "\n")

    # 选取 4 个典型样本分散对比展示
    sample_indices = [0, limit // 3, 2 * limit // 3, max(0, limit - 1)]
    # 去重且保持顺序
    seen = set()
    unique_indices = [x for x in sample_indices if not (x in seen or seen.add(x))]
    case_studies = []
    for idx in unique_indices:
        case = {
            "index": idx,
            "emotion": emotions[idx],
            "context": contexts[idx][-2:] if len(contexts[idx]) >= 2 else contexts[idx],
            "reference": references[idx],
            "outputs": {k: strategies[k][idx] for k in strategies},
        }
        case_studies.append(case)

    return results_summary, case_studies, strategies


def main():
    parser = argparse.ArgumentParser(description="Diverse Adaptive Decoding Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="save/epcl_v6_trial3/CASE_39999_36.8639",
        help="Checkpoint path for evaluation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of test samples to evaluate (default 500 for fast verification, 0 for all)",
    )
    parser.add_argument(
        "--max_dec_step",
        type=int,
        default=50,
        help="Maximum decoding steps",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/route_a_decoding_eval.json",
        help="Output JSON file path",
    )
    args, _ = parser.parse_known_args()

    model, vocab, test_set, translator = load_eval_model(args.checkpoint)
    summary, cases, _ = run_evaluation(
        model, vocab, test_set, translator, num_samples=args.num_samples, max_dec_step=args.max_dec_step
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_data = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples if args.num_samples > 0 else len(test_set),
        "summary": summary,
        "case_studies": cases,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 评测指标与定性案例已保存至: {args.output_file}")

    # 同时导出 Markdown 格式总结报告
    md_file = args.output_file.replace(".json", ".md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# 路线 A: 多样性自适应解码评测结果 (Diverse Adaptive Decoding Report)\n\n")
        f.write(f"- **模型权重**: `{args.checkpoint}`\n")
        f.write(f"- **评测样本数**: {out_data['num_samples']}\n\n")
        f.write("## 1. 多样性量化指标对比表\n\n")
        f.write("| 解码策略 (Strategy) | Dist-1 (%) | Dist-2 (%) | Unique Sentences (%) | Avg Length |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for strat, m in summary.items():
            f.write(f"| **{strat}** | {m['Dist-1']:.2f}% | {m['Dist-2']:.2f}% | {m['Unique']:.2f}% | {m['Avg_Len']:.2f} |\n")
        f.write("\n## 2. 典型样本定性对比 (Case Studies)\n\n")
        for c in cases:
            f.write(f"### 样本 #{c['index']} [情感: {c['emotion']}]\n")
            f.write(f"- **上下文 (Context)**: {' | '.join(c['context'])}\n")
            f.write(f"- **人类真实回复 (Ground Truth)**: {c['reference']}\n")
            f.write("- **各策略模型生成回复**:\n")
            for strat, text in c['outputs'].items():
                f.write(f"  - **{strat}**: {text}\n")
            f.write("\n")
    print(f"[OK] Markdown 评测报告已保存至: {md_file}")


if __name__ == "__main__":
    main()
