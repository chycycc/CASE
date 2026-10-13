# -*- coding: utf-8 -*-
"""
CASE-EPCL V9 Trial 10: 阶段 0 PCAM 架构动力学离线诊断脚本 (diag_pcam_abc.py 终极完善版)

【落地防御与增强清单】:
  - H1 关键权重严格断言: 强制断言 pcam/epcl_criterion/emotion_linear 权重非空且无未匹配项;
  - H2 NLTK 口径严格对齐: 复用 word_tokenize 与 tuple 集合，与 eval_pipeline 100% 同构;
  - H3 严谨自检 (N2 收紧至 ±0.08): 同时校准 Dev PPL (37.7145) 与 Test PPL (32.7836)，区分 Trial 8/9;
  - N3 熔断数据保全: 即使自检超差，亦优先完成 JSON 序列化与紧凑打印，绝不丢失 C/D/E/C' 诊断数据;
  - N4 修正命令行参数文本: --ignore_self_check 明确为“自检超差仅警告而不强制退出”;
  - N5 LN 仿射统计全量进 JSON: report() 完整包含 mean/std/min/max;
  - E1 Trial 8 历史锚点对照 (32.97): 模式 C (h_t 直通) 直接对标 Trial 8 推理路径;
  - E2 2x2 析因交互分析: 计算 Δ 效应、LN 效应与非线性交互项交互调制;
  - 性能隐患根治: 循环外单次 word_tokenize，重采样秒级完成;
  - 零文件传输优化: 终端单行打印 [DIAG_PCAM_COMPACT_JSON]。
"""

import os
import sys
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from nltk import word_tokenize

# 项目路径注入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq


def calc_distinct_2_from_tokens(tok_list):
    """基于预分词 token 列表的高性能 Bigram 多样性计算"""
    total = 0
    d = set()
    for ws in tok_list:
        for i in range(len(ws) - 1):
            d.add((ws[i], ws[i + 1]))
            total += 1
    return (len(d) / (total + 1e-16)) * 100.0


def calc_distinct_1_from_tokens(tok_list):
    """基于预分词 token 列表的 Unigram 多样性计算"""
    total = 0
    d = set()
    for ws in tok_list:
        for w in ws:
            d.add(w)
            total += 1
    return (len(d) / (total + 1e-16)) * 100.0


def paired_bootstrap_diff(toks_a, toks_b, n_boot=1000):
    """
    高性能配对 Bootstrap 检验 (预分词单次处理，秒级重采样)
    """
    n = len(toks_a)
    assert n == len(toks_b) and n > 0, "样本列表长度必须一致且非空"
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample_a = [toks_a[i] for i in idx]
        sample_b = [toks_b[i] for i in idx]
        d2_a = calc_distinct_2_from_tokens(sample_a)
        d2_b = calc_distinct_2_from_tokens(sample_b)
        diffs.append(d2_a - d2_b)
    mean_diff = float(np.mean(diffs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return mean_diff, (float(lo), float(hi))


class ScaleStatsCollector:
    """模式 S: 隐状态物理尺度统计器"""
    def __init__(self, ln_weight=None):
        self.ln_weight = ln_weight.detach().cpu().numpy() if ln_weight is not None else None
        self.reset()

    def reset(self):
        self.h_t_norms = []
        self.h_t_stds = []
        self.ln_out_stds = []
        self.perturb_norms = []
        self.energy_ratios = []
        self.gate_means = []

    def record(self, dec_output, perturbation, gate, ln_out):
        with torch.no_grad():
            d_norm = torch.norm(dec_output, p=2, dim=-1)
            p_norm = torch.norm(perturbation, p=2, dim=-1)
            d_std = torch.std(dec_output, dim=-1)
            l_std = torch.std(ln_out, dim=-1)
            ratio = p_norm / (d_norm + 1e-8)

            self.h_t_norms.append(d_norm.mean().item())
            self.h_t_stds.append(d_std.mean().item())
            self.ln_out_stds.append(l_std.mean().item())
            self.perturb_norms.append(p_norm.mean().item())
            self.energy_ratios.append(ratio.mean().item())
            self.gate_means.append(gate.mean().item())

    def report(self):
        print("\n" + "=" * 70)
        print(" [模式 S: 隐状态与扰动尺度物理统计诊断报告]")
        print("=" * 70)
        mean_energy = float(np.mean(self.energy_ratios) * 100)
        mean_ht_std = float(np.mean(self.h_t_stds))
        mean_ln_std = float(np.mean(self.ln_out_stds))
        mean_gate = float(np.mean(self.gate_means))

        print(f" 1. 解码主干隐状态 h_t 平均 L2 模长:   {np.mean(self.h_t_norms):.4f} ± {np.std(self.h_t_norms):.4f}")
        print(f" 2. 解码主干隐状态 h_t 各维实测标准差: {mean_ht_std:.4f} ± {np.std(self.h_t_stds):.4f}")
        print(f" 3. LayerNorm 输出经重标定后标准差:    {mean_ln_std:.4f} ± {np.std(self.ln_out_stds):.4f}")
        print(f" 4. 微扰能量比 (||Δh|| / ||h_t||):     {mean_energy:.2f}% ± {np.std(self.energy_ratios) * 100:.2f}%")
        print(f" 5. 通道自适应门控 gate 激活均值:      {mean_gate:.4f} ± {np.std(self.gate_means):.4f}")

        w_summary = None
        if self.ln_weight is not None:
            w_mean = float(np.mean(self.ln_weight))
            w_min = float(np.min(self.ln_weight))
            w_max = float(np.max(self.ln_weight))
            w_std = float(np.std(self.ln_weight))
            w_summary = {"mean": round(w_mean, 4), "std": round(w_std, 4), "min": round(w_min, 4), "max": round(w_max, 4)}
            print(f" 6. LayerNorm 仿射权重 weight:         均值={w_mean:.4f}, 方差={w_std:.4f}, 极值=[{w_min:.4f}, {w_max:.4f}]")
            if abs(w_mean - 1.0) > 0.15 or w_std > 0.2:
                print("    --> [物理诊断]: LN weight 显著偏离各向同性 1.0！收益强依赖仿射重标定。")
            else:
                print("    --> [物理诊断]: LN weight 接近单位初始化，收益主要来自方差重归一化。")

        print("-" * 70)
        # N5 完整输出返回字典
        return {
            "mean_energy_pct": round(mean_energy, 4),
            "mean_ht_std": round(mean_ht_std, 4),
            "mean_ln_std": round(mean_ln_std, 4),
            "mean_gate": round(mean_gate, 4),
            "ln_weight": w_summary
        }


def build_patched_forward(mode='A', collector=None, c_prime_ln=None):
    def patched_forward(self, dec_output, prototypes):
        bsz, seq_len, _ = dec_output.size()

        q = self.q_proj(dec_output) * self.scaling
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if prototypes.dim() == 2:
            k_protos, _ = prototypes.size()
            k = self.k_proj(prototypes).view(k_protos, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
            v = self.v_proj(prototypes).view(k_protos, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
        else:
            _, k_protos, _ = prototypes.size()
            k = self.k_proj(prototypes).view(bsz, k_protos, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(prototypes).view(bsz, k_protos, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        h_proto = self.out_dropout(self.out_proj(context))

        gate_input = torch.cat([dec_output, h_proto], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        delta = torch.tanh(self.gamma) * (gate * h_proto)

        if collector is not None:
            ln_sample = self.layer_norm(dec_output + delta)
            collector.record(dec_output, delta, gate, ln_sample)

        if mode == 'A':
            return self.layer_norm(dec_output + delta)
        elif mode == 'B':
            return self.layer_norm(dec_output)
        elif mode == 'C':
            return dec_output
        elif mode == 'D':
            return dec_output + delta
        elif mode == 'E':
            return dec_output + torch.tanh(self.gamma) * (gate * self.layer_norm(h_proto))
        elif mode == "C'":
            bypass_branch = torch.tanh(self.gamma) * (gate * self.layer_norm(h_proto))
            return c_prime_ln(dec_output + bypass_branch)
        else:
            raise ValueError(f"未知评估模式: {mode}")

    return patched_forward


def evaluate_pure_forward_ppl(model, dataloader, desc="Evaluating"):
    model.eval()
    ctx_loss_list = []
    emo_acc_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            _, _, _, ctx, _, _, _, _, emo_acc, _, _ = model.train_one_batch(
                batch, 0, train=False
            )
            ctx_loss_list.append(ctx)
            emo_acc_list.append(emo_acc)

    mean_ctx = np.mean(ctx_loss_list)
    overall_ppl = math.exp(mean_ctx)
    mean_emo_acc = np.mean(emo_acc_list) * 100.0
    return overall_ppl, mean_emo_acc


def run_greedy_decode_tokens(model, test_set_single, max_samples=500):
    """生成多样性采样并执行预分词 (单次 word_tokenize，彻底消除百万次循环调用)"""
    model.eval()
    tokenized_cands = []
    raw_cands = []
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_set_single, desc="Greedy Decoding", total=max_samples, leave=False):
            if count >= max_samples:
                break
            count += 1
            sent_g = model.decoder_greedy(batch, max_dec_step=40)
            cand_str = sent_g[0]
            raw_cands.append(cand_str)
            tokenized_cands.append(word_tokenize(cand_str))

    return tokenized_cands, raw_cands


def safe_load_checkpoint(checkpoint_path, batch_size=32):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    print(f"\n[*] 预检检查点拓扑结构: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    raw_keys = list(state["model"].keys() if "model" in state else state.keys())

    # H1 关键前缀断言
    assert any(k.startswith("pcam.") for k in raw_keys), f"[致命 H1] 检查点不含 PCAM 权重: {checkpoint_path}"
    assert any(k.startswith("epcl_criterion.") for k in raw_keys), f"[致命 H1] 检查点不含 EPCL 权重: {checkpoint_path}"
    assert any(k.startswith("emotion_linear.") for k in raw_keys), f"[致命 H1] 检查点不含 分类头 权重: {checkpoint_path}"

    if any("emotion_linear.linear_2." in k or "emotion_linear.block." in k for k in raw_keys):
        config.emotion_head_type = "residual_mlp"
        print("    [拓扑识别] 检出残差 MLP 分类头 (residual_mlp)")
    else:
        config.emotion_head_type = "linear"
        print("    [拓扑识别] 检出标准线性分类头 (linear)")

    config.use_pcam = True
    config.use_erp = any("erp." in k for k in raw_keys)
    config.dataset = "ED"
    config.batch_size = batch_size
    config.woStrategy = True
    config.mlp_hidden_dim = 300
    config.mlp_dropout = 0.1
    config.num_prototypes_per_class = 1
    config.pcam_heads = 2
    config.pcam_dropout = 0.1
    config.pcam_gate_bias = -1.0
    config.pcam_tau_attn = 1.5
    config.use_mcp = True
    config.mcp_momentum = 0.96
    config.epcl_anchor = "fine_emotion"
    config.cls_anchor = "default"
    config.test = True
    config.model = "case"
    config.save_path = os.path.join(PROJECT_ROOT, "save", "diag_cache")
    os.makedirs(config.save_path, exist_ok=True)

    if not hasattr(config, "device") or not config.device:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed()

    print(f"[*] 加载 EmpatheticDialogues 数据集 (Batch Size: {batch_size})...")
    train_set, dev_set, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=batch_size)

    model = CASE(
        vocab,
        emotion_num=emo_num,
        strategy_num=strategy_num,
        is_eval=True,
        model_file_path=None,
    ).to(config.device)

    model_state = state["model"] if "model" in state else state
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

    # H1 关键权重不可缺失与错配防御
    CRITICAL = ("pcam.", "epcl_criterion.", "emotion_linear.", "decoder.", "encoder.", "generator.")
    bad_missing = [k for k in missing_keys if any(k.startswith(p) for p in CRITICAL)]
    bad_unexpected = [k for k in unexpected_keys if any(k.startswith(p) for p in CRITICAL)]
    assert not bad_missing, f"[致命 H1] 关键权重缺失，禁止继续评测: {bad_missing[:10]}"
    assert not bad_unexpected, f"[致命 H1] 关键权重未匹配，拓扑装配错误: {bad_unexpected[:10]}"

    model.eval()
    model.is_eval = True
    print("[+] H1 拓扑断言通过，权重加载就绪！")
    return model, vocab, dev_set, test_set


def run_diagnostics(args):
    print("=" * 75)
    print(" CASE-EPCL V9 Trial 10: 阶段 0 PCAM 架构动力学离线诊断 (终极完善封版)")
    print("=" * 75)

    model, vocab, dev_set, test_set = safe_load_checkpoint(args.model_path, batch_size=args.batch_size)

    gamma_val = model.pcam.gamma.item()
    tanh_gamma = math.tanh(gamma_val)
    print(f"[*] 检出 PCAM 参数: gamma = {gamma_val:.6f}, tanh(gamma) = {tanh_gamma:.6f}")

    c_prime_ln_out = nn.LayerNorm(300).to(config.device)

    # ================= 诊断 1: 模式 S 物理尺度实测 =================
    collector = ScaleStatsCollector(ln_weight=getattr(model.pcam.layer_norm, 'weight', None))
    original_forward = model.pcam.__class__.forward
    model.pcam.__class__.forward = build_patched_forward(mode='A', collector=collector, c_prime_ln=c_prime_ln_out)

    print("\n[*] [阶段 0.1] 正在测试集上执行模式 S 物理尺度采集...")
    with torch.no_grad():
        evaluate_pure_forward_ppl(model, test_set, desc="Scale Measuring")
    scale_summary = collector.report()

    # ================= 诊断 2: 模式 A/B/C/D/E/C' 纯前向对比 =================
    modes = ['A', 'B', 'C', 'D', 'E', "C'"]
    mode_descs = {
        'A': 'LN(h_t + Δ) [原版基线]',
        'B': 'LN(h_t) [断开微扰]',
        'C': 'h_t [真·去 LN 承重墙, 对标 Trial 8]',
        'D': 'h_t + Δ [去 LN 保 Δ]',
        'E': 'h_t + tanh(γ)·(gate·LN(h_proto)) [proxy, OOD 下界]',
        "C'": 'LN_out(h_t + tanh(γ)·(gate·LN(h_proto))) [proxy, 未训练 LN_out]'
    }

    ppl_results = {}
    print("=" * 75)
    print(" [*] [阶段 0.2] 开始执行 6 大物理模式纯前向 PPL 严谨度量")
    print("=" * 75)

    model.pcam.__class__.forward = build_patched_forward(mode='A', collector=None, c_prime_ln=c_prime_ln_out)
    dev_ppl_a, dev_emo_acc_a = evaluate_pure_forward_ppl(model, dev_set, desc="Mode A (Dev)")

    for m in modes:
        model.pcam.__class__.forward = build_patched_forward(mode=m, collector=None, c_prime_ln=c_prime_ln_out)
        ppl_val, emo_acc_val = evaluate_pure_forward_ppl(model, test_set, desc=f"Mode {m}")
        ppl_results[m] = {
            "test_ppl": round(ppl_val, 4),
            "test_emo_acc": round(emo_acc_val, 2),
            "desc": mode_descs[m]
        }
        print(f"  --> 模式 {m:<3} | Test PPL: {ppl_val:8.4f} | EMO Acc: {emo_acc_val:5.2f}% | {mode_descs[m]}")

    # ================= E2: 2x2 析因交互分析 =================
    ppl_a = ppl_results['A']['test_ppl']
    ppl_b = ppl_results['B']['test_ppl']
    ppl_c = ppl_results['C']['test_ppl']
    ppl_d = ppl_results['D']['test_ppl']

    delta_effect_ln = ppl_a - ppl_b       # LN 域内 Δ 的效应
    delta_effect_bare = ppl_d - ppl_c     # 无 LN 域内 Δ 的效应
    ln_effect_zero = ppl_b - ppl_c        # 纯 LN 效应 (无 Δ)
    interaction_effect = delta_effect_ln - delta_effect_bare  # 交互项

    factorial_analysis = {
        "delta_effect_in_ln": round(delta_effect_ln, 4),
        "delta_effect_bare": round(delta_effect_bare, 4),
        "ln_effect_pure": round(ln_effect_zero, 4),
        "interaction_effect": round(interaction_effect, 4)
    }

    # ================= 诊断 3: 因果多样性解码与高性能 M2 配对 Bootstrap =================
    decode_summary = {}
    if not args.no_decode:
        print("\n" + "=" * 75)
        print(f" [*] [阶段 0.3] 执行前 {args.decode_samples} 句解码 (模式 A vs B) + M2 高性能配对 Bootstrap")
        print("=" * 75)
        _, _, test_set_single, _, _, _ = prepare_data_seq(batch_size=1)

        model.pcam.__class__.forward = build_patched_forward(mode='A', collector=None, c_prime_ln=c_prime_ln_out)
        toks_a, _ = run_greedy_decode_tokens(model, test_set_single, max_samples=args.decode_samples)

        model.pcam.__class__.forward = build_patched_forward(mode='B', collector=None, c_prime_ln=c_prime_ln_out)
        toks_b, _ = run_greedy_decode_tokens(model, test_set_single, max_samples=args.decode_samples)

        d1_a = calc_distinct_1_from_tokens(toks_a)
        d2_a = calc_distinct_2_from_tokens(toks_a)
        d1_b = calc_distinct_1_from_tokens(toks_b)
        d2_b = calc_distinct_2_from_tokens(toks_b)

        mean_diff, (diff_lo, diff_hi) = paired_bootstrap_diff(toks_a, toks_b, n_boot=1000)
        contains_zero = (diff_lo <= 0.0 <= diff_hi)

        decode_summary = {
            "mode_A": {"dist1": round(d1_a, 2), "dist2": round(d2_a, 2)},
            "mode_B": {"dist1": round(d1_b, 2), "dist2": round(d2_b, 2)},
            "paired_bootstrap": {
                "mean_delta_dist2": round(mean_diff, 4),
                "ci95": [round(diff_lo, 4), round(diff_hi, 4)],
                "contains_zero": contains_zero,
                "is_significant": not contains_zero
            }
        }

        print(f"  模式 A Dist-2: {d2_a:.2f}% | 模式 B Dist-2: {d2_b:.2f}%")
        print(f"  [M2 配对检验]: Δ Dist-2 均值 = {mean_diff:+.4f}%, 95% CI = [{diff_lo:+.4f}%, {diff_hi:+.4f}%]")
        if contains_zero:
            print("  --> [因果结论]: 95% CI 包含 0！原型微扰对生成多样性【无统计显著因果驱动】。")
        else:
            print("  --> [因果结论]: 95% CI 不包含 0！原型微扰对生成多样性存在显著正向驱动。")

    # ================= 产物结构化落盘 (N3: 熔断前优先落盘) =================
    output_dir = os.path.join(PROJECT_ROOT, "save", "diag")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "pcam_diag_results.json")

    # H3 锚点吻合性校验 (N2: 收紧至 ±0.08)
    dev_err = abs(dev_ppl_a - 37.7145)
    test_err = abs(ppl_a - 32.7836)
    self_check_passed = bool(dev_err <= 0.08 and test_err <= 0.08)

    final_report = {
        "checkpoint": args.model_path,
        "gamma": gamma_val,
        "tanh_gamma": tanh_gamma,
        "dev_ppl_baseline": round(dev_ppl_a, 4),
        "self_check": {
            "passed": self_check_passed,
            "dev_err": round(dev_err, 4),
            "test_err": round(test_err, 4),
            "tolerance": 0.08
        },
        "factorial_analysis": factorial_analysis,
        "scale_stats": scale_summary,
        "ppl_comparison": ppl_results,
        "decode_diversity": decode_summary
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    model.pcam.__class__.forward = original_forward

    # 零传输摩擦: 紧凑 JSON 单行打印
    print("\n" + "=" * 75)
    print(" [DIAG_PCAM_COMPACT_JSON]")
    print(json.dumps(final_report, ensure_ascii=False))
    print("=" * 75 + "\n")

    # ================= E1: 历史锚点与析因总结输出 =================
    print("=" * 75)
    print("                       阶段 0 离线诊断核心判读结论")
    print("=" * 75)
    print(f" 1. 历史基线对标 (E1): 模式 C (h_t 直通) PPL = {ppl_c:.4f} vs Trial 8 (32.97)")
    if ppl_c <= 33.15:
        print("    --> [重大判读]: 模式 C 逼近 32.97！外层 LN 非核心承重墙，原 0.18 提升并非源于 LN，原恒等重构失去必要！")
    else:
        print(f"    --> [重大判读]: 模式 C 显著恶化 (Δ = {ppl_c - ppl_a:+.4f})！外层 LN 确为承重墙，C' 两全方案优先级最高。")

    print(f" 2. 2x2 析因交互 (E2): Δ 效应 (LN 域内) = {delta_effect_ln:+.4f} | Δ 效应 (裸残差) = {delta_effect_bare:+.4f} | 交互项 = {interaction_effect:+.4f}")

    # N3 熔断判定 (在完整输出之后执行)
    if not self_check_passed:
        print(f"\n[!] 警告: H3 锚点自检超出 ±0.08 阈值 (Dev 差={dev_err:.4f}, Test 差={test_err:.4f})！")
        if not args.ignore_self_check:
            print("    [!] 触发保护性熔断退出。若确认环境差异可添加 --ignore_self_check 忽略。")
            sys.exit(1)
    else:
        print(" [+] H3 双锚点自检完美通过 (偏差均 <= 0.08)！流水线数据 100% 真实可信。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CASE-EPCL PCAM 阶段 0 动力学离线诊断脚本")
    parser.add_argument("--model_path", type=str, default="save/v9_trial9/CASE_7499_37.7145_composite_best", help="目标检查点路径")
    parser.add_argument("--batch_size", type=int, default=32, help="纯前向评估批次大小")
    parser.add_argument("--no_decode", action="store_true", help="显式关闭因果多样性解码检验")
    parser.add_argument("--decode_samples", type=int, default=500, help="因果解码抽样样本数")
    parser.add_argument("--ignore_self_check", action="store_true", help="自检超差时仅警告而不强制退出")
    args = parser.parse_args()

    run_diagnostics(args)
