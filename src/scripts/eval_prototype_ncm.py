# -*- coding: utf-8 -*-
"""
CASE-EPCL 学术评测流水线: 原型最近邻分类器 (NCM) 评测、逐样本表征落盘与 McNemar 配对检验脚本
脚本路径: src/scripts/eval_prototype_ncm.py (由 scratch/eval_dpfc_offline.py 升格)

【核心功能】:
1. 权重拓扑自适应加载与关键断言 (H1 防御):
   - 自动检测并断言 pcam / epcl_criterion / emotion_linear / residual_mlp 权重;
2. 测试集/验证集特征与 Logits 提取:
   - 使用 Hook 捕获真实样本在编码器输出端的投影隐向量与分类头 Logits;
   - 强一致性防御 (L1): 逐批核对 Hook 提取预测与模型内置预测;
3. 原型空间最近质心/均值分类 (NCM):
   - 计算余弦相似度与温度尺度变换下的类别指派与准确率;
4. 离线表征矩阵与概率深度落盘:
   - 导出 dpfc_preds.npz (包含 true_labels, linear_probs, ncm_sim, preds_linear, preds_ncm, preds_dpfc);
   - 支撑跨 Trial (如 T12 vs T11-B, T12 vs T12-C) 的离线配对 McNemar 检验;
5. 显著性检验与结构化报告:
   - 运行连续性校正 McNemar 检验与双侧精确二项检验，输出标准化 JSON 报告。
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

# 项目根目录路径注入
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq


def safe_load_checkpoint(checkpoint_path, batch_size=32):
    """
    自适应定位并安全加载模型检查点，执行拓扑识别与关键权重断言 (H1 防御)。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        candidates = [
            os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path)
            if not f.endswith((".json", ".txt", ".png", ".tmp", ".log")) and os.path.isfile(os.path.join(checkpoint_path, f))
        ]
        composite_cands = [f for f in candidates if "composite_best" in os.path.basename(f)]
        if composite_cands:
            checkpoint_path = composite_cands[0]
        elif candidates:
            case_cands = [f for f in candidates if "CASE_" in os.path.basename(f)]
            checkpoint_path = case_cands[0] if case_cands else candidates[0]
        else:
            raise FileNotFoundError(f"在目录中未找到有效模型检查点文件: {checkpoint_path}")
        print(f"[*] 目录输入自动定位目标检查点: {checkpoint_path}")

    print(f"\n[*] 预检检查点拓扑结构: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    raw_keys = list(state["model"].keys() if "model" in state else state.keys())

    # H1 关键前缀断言 (自适应 PCAM / Baseline)
    has_pcam = any(k.startswith("pcam.") for k in raw_keys)
    config.use_pcam = has_pcam
    if has_pcam:
        print("    [拓扑识别] 检出 PCAM 特征增强模块")
    else:
        print("    [拓扑识别] 无 PCAM 模块 (标准 Baseline 模式)")

    assert any(k.startswith("epcl_criterion.") for k in raw_keys), f"[致命 H1] 检查点不含 EPCL 权重: {checkpoint_path}"
    assert any(k.startswith("emotion_linear.") for k in raw_keys), f"[致命 H1] 检查点不含分类头权重: {checkpoint_path}"

    if any("emotion_linear.linear_2." in k or "emotion_linear.block." in k for k in raw_keys):
        config.emotion_head_type = "residual_mlp"
        print("    [拓扑识别] 检出残差 MLP 分类头 (residual_mlp)")
    else:
        config.emotion_head_type = "linear"
        print("    [拓扑识别] 检出标准线性分类头 (linear)")

    # T12 实际参数键为 decoder.dec.0.prefix_embed；必须同步重置关闭状态。
    prefix_keys = [k for k in raw_keys if k.endswith("prefix_embed")]
    config.use_proto_prefix = bool(prefix_keys)
    if prefix_keys:
        model_state = state["model"] if "model" in state else state
        config.proto_prefix_len = model_state[prefix_keys[0]].shape[0]
    print(f"    [拓扑识别] use_proto_prefix={config.use_proto_prefix}"
          f"{', P=' + str(config.proto_prefix_len) if prefix_keys else ''}")

    gate_keys = [k for k in raw_keys if k.endswith("prefix_gate")]
    if gate_keys and not prefix_keys:
        raise RuntimeError("Gate weights found without prefix embeddings")
    declared_mode = state.get("prefix_attention_mode")
    if declared_mode is not None:
        if declared_mode not in ("none", "legacy_joint_softmax", "separate_softmax_residual"):
            raise RuntimeError("Unknown checkpoint prefix_attention_mode")
        if bool(gate_keys) != (declared_mode == "separate_softmax_residual") or bool(prefix_keys) != (declared_mode != "none"):
            raise RuntimeError("Checkpoint prefix metadata/weights disagree; refusing topology fallback")
    config.use_proto_prefix_gate = bool(gate_keys)
    print(f"    [拓扑识别] use_proto_prefix_gate={config.use_proto_prefix_gate}")
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

    CRITICAL = ("pcam.", "epcl_criterion.", "emotion_linear.", "decoder.", "encoder.", "generator.")
    bad_missing = [k for k in missing_keys if any(k.startswith(p) for p in CRITICAL)]
    bad_unexpected = [k for k in unexpected_keys if any(k.startswith(p) for p in CRITICAL)]
    assert not bad_missing, f"[致命 H1] 关键权重缺失，禁止继续评测: {bad_missing[:10]}"
    assert not bad_unexpected, f"[致命 H1] 关键权重未匹配，拓扑装配错误: {bad_unexpected[:10]}"

    model.eval()
    model.is_eval = True
    model.print_proto_prefix_gate_status(stage="checkpoint_restored")
    print("[+] H1 拓扑断言通过，权重加载就绪！")
    return model, vocab, dev_set, test_set


def extract_features_and_logits(model, dataloader, split_name="dev"):
    """
    通过注册 Hook 提取特征表征与分类头输出 logits。
    """
    model.eval()
    captured_logits = []

    def hook_fn(module, input, output):
        captured_logits.append(output.detach())

    hook_handle = model.emotion_linear.register_forward_hook(hook_fn)

    all_logits = []
    all_proj_feats = []
    all_labels = []

    print(f"[*] 正在提取 {split_name} 集特征与 Logits (共 {len(dataloader)} 批次)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split_name}"):
            captured_logits.clear()
            model.train_one_batch(batch, 0, train=False)

            if len(captured_logits) == 0:
                raise RuntimeError(f"Hook 未捕获到分类头输出！")

            batch_logits = captured_logits[0].clone()
            captured_logits.clear()

            if hasattr(model, 'current_pred_emotion') and model.current_pred_emotion is not None:
                hook_preds = batch_logits.argmax(dim=-1).cpu().numpy()
                assert np.array_equal(hook_preds, model.current_pred_emotion), \
                    f"[致命 L1] Hook 提取与模型内置 current_pred_emotion 错位！"

            if hasattr(model, 'current_projected_emotion') and model.current_projected_emotion is not None:
                batch_proj = model.current_projected_emotion.detach()
            else:
                raise RuntimeError("未检出 model.current_projected_emotion 投影记录！")

            batch_labels = batch["program_label"].detach()

            all_logits.append(batch_logits.cpu())
            all_proj_feats.append(batch_proj.cpu())
            all_labels.append(batch_labels.cpu())

    hook_handle.remove()

    logits_tensor = torch.cat(all_logits, dim=0)
    proj_tensor = torch.cat(all_proj_feats, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    print(f"[*] {split_name} 集提取完成: 样本数={logits_tensor.size(0)}, 类别数={logits_tensor.size(1)}, 投影维度={proj_tensor.size(1)}")
    return logits_tensor, proj_tensor, labels_tensor


def compute_ncm_logits(proj_norm, proto_norm, valid_mask=None, num_prototypes_per_class=1):
    """
    计算 L2 归一化特征与原型矩阵的余弦相似度。
    """
    N, D = proj_norm.size()
    total_protos, _ = proto_norm.size()
    num_classes = total_protos // num_prototypes_per_class

    sim = torch.matmul(proj_norm, proto_norm.T)

    if valid_mask is not None:
        invalid_mask = (~valid_mask).unsqueeze(0).expand(N, -1)
        sim = sim.masked_fill(invalid_mask, -1e9)

    if num_prototypes_per_class > 1:
        sim_reshaped = sim.view(N, num_classes, num_prototypes_per_class)
        sim_class, _ = sim_reshaped.max(dim=-1)
    else:
        sim_class = sim

    return sim_class


def evaluate_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return (correct / float(total)) * 100.0, preds


def grid_search_dpfc(dev_logits, dev_proj, dev_labels, proto_norm, valid_mask, num_protos_per_class, tau_list, alpha_list):
    """
    在验证集上执行网格寻优，输出 NCM 最优 tau 与后验融合最优参数（用于历史对照与机制闭环）。
    """
    print("\n[*] 正在验证集上执行独立参数寻优...")

    # 1. NCM-Only 独立最优寻优
    best_ncm_acc = -1.0
    best_ncm_tau = None
    for tau in tau_list:
        ncm_sim = compute_ncm_logits(dev_proj, proto_norm, valid_mask, num_protos_per_class)
        ncm_logits = ncm_sim / tau
        acc, _ = evaluate_accuracy(ncm_logits, dev_labels)
        if acc > best_ncm_acc:
            best_ncm_acc = acc
            best_ncm_tau = tau

    print(f"[*] NCM-Only 独立最优参数: tau* = {best_ncm_tau}, Dev Acc = {best_ncm_acc:.2f}%")

    # 2. DPFC 联合网格寻优
    best_dpfc_acc = -1.0
    best_dpfc_params = {}
    linear_probs = F.softmax(dev_logits, dim=-1)

    for tau in tau_list:
        ncm_sim = compute_ncm_logits(dev_proj, proto_norm, valid_mask, num_protos_per_class)
        ncm_logits = ncm_sim / tau
        ncm_probs = F.softmax(ncm_logits, dim=-1)

        for alpha in alpha_list:
            fused_probs = (1.0 - alpha) * linear_probs + alpha * ncm_probs
            acc_prob, _ = evaluate_accuracy(fused_probs, dev_labels)
            if acc_prob > best_dpfc_acc:
                best_dpfc_acc = acc_prob
                best_dpfc_params = {'tau': tau, 'alpha': alpha, 'type': 'prob', 'acc': acc_prob}

            fused_logits = (1.0 - alpha) * dev_logits + alpha * ncm_logits
            acc_logit, _ = evaluate_accuracy(fused_logits, dev_labels)
            if acc_logit > best_dpfc_acc:
                best_dpfc_acc = acc_logit
                best_dpfc_params = {'tau': tau, 'alpha': alpha, 'type': 'logit', 'acc': acc_logit}

    print(f"[*] DPFC 融合最优参数: 模式={best_dpfc_params['type']}, tau*={best_dpfc_params['tau']}, alpha*={best_dpfc_params['alpha']}, Dev Acc={best_dpfc_params['acc']:.2f}%")
    return best_ncm_tau, round(best_ncm_acc, 2), best_dpfc_params


def exact_binomial_test_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    prob_sum = 0.0
    for i in range(0, k + 1):
        prob_sum += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(1.0, 2.0 * prob_sum)


def mcnemar_test(b, c):
    """
    McNemar 配对卡方检验（含 Yates 连续性校正）。
    """
    total = b + c
    if total == 0:
        return 0.0, 1.0, "Identical"

    if total < 25:
        p_value = exact_binomial_test_two_sided(min(b, c), total)
        return float(abs(b - c)), p_value, "Exact Binomial"
    else:
        chi2 = (abs(b - c) - 1.0) ** 2 / float(total)
        p_value = math.erfc(math.sqrt(chi2 / 2.0))
        return chi2, p_value, "Chi-Square (continuity corrected)"


def run_ncm_evaluation(args):
    print("=" * 75)
    print(" CASE-EPCL 学术评测流水线: 原型空间 (NCM) 评测与逐样本矩阵落盘")
    print("=" * 75)

    model, vocab, dev_set, test_set = safe_load_checkpoint(args.model_path, batch_size=args.batch_size)

    num_protos_per_class = getattr(model, 'num_prototypes_per_class', 1)
    raw_protos = model.epcl_criterion.prototypes.detach().cpu()
    proto_norm = F.normalize(raw_protos, p=2, dim=-1)

    total_protos = proto_norm.size(0)
    # N1: 降级断言为警告与掩码保护，保证未完全初始化的检查点安全运转
    if hasattr(model.epcl_criterion, 'class_initialized'):
        valid_mask = model.epcl_criterion.class_initialized.detach().cpu().bool()
        num_init = int(valid_mask.sum().item())
        print(f"[*] 检出 class_initialized buffer: 已初始化类别数={num_init}/{total_protos}")
        if num_init < total_protos:
            print(f"[!] 警告 (N1): 存在 {total_protos - num_init} 个类别原型未被初始化，已自动启用 -1e9 掩码屏蔽。")
    else:
        valid_mask = (torch.norm(raw_protos, p=2, dim=-1) > 1e-4)
        num_init = int(valid_mask.sum().item())
        print(f"[*] 依据模长检测有效原型: 有效原型数={num_init}/{total_protos}")

    # 特征与 Logits 提取 (含 L1 强一致性防御)
    dev_logits, dev_proj, dev_labels = extract_features_and_logits(model, dev_set, split_name="Dev")
    test_logits, test_proj, test_labels = extract_features_and_logits(model, test_set, split_name="Test")

    # 基准评估: Linear Only
    dev_linear_acc, _ = evaluate_accuracy(dev_logits, dev_labels)
    test_linear_acc, test_linear_preds = evaluate_accuracy(test_logits, test_labels)

    print("\n" + "=" * 70)
    print(" [基准参考: 纯 Linear 分类头 (Alpha = 0.0)]")
    print(f"   - Dev Set Acc:  {dev_linear_acc:.2f}%")
    print(f"   - Test Set Acc: {test_linear_acc:.2f}%")
    print("=" * 70)

    # 网格搜索
    tau_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    alpha_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    best_ncm_tau, dev_ncm_acc, best_dpfc_params = grid_search_dpfc(
        dev_logits, dev_proj, dev_labels, proto_norm, valid_mask, num_protos_per_class, tau_list, alpha_list
    )
    dev_dpfc_acc = best_dpfc_params['acc']

    # 测试集冻结参数评测
    # 1. 纯 NCM 评测 (专属最优 tau)
    test_ncm_sim = compute_ncm_logits(test_proj, proto_norm, valid_mask, num_protos_per_class)
    test_ncm_logits = test_ncm_sim / best_ncm_tau
    test_ncm_acc, test_ncm_preds = evaluate_accuracy(test_ncm_logits, test_labels)

    # 2. DPFC 融合评测 (联合最优超参)
    tau_star = best_dpfc_params['tau']
    alpha_star = best_dpfc_params['alpha']
    f_type = best_dpfc_params['type']

    fused_ncm_logits = test_ncm_sim / tau_star
    test_linear_probs = F.softmax(test_logits, dim=-1)
    test_ncm_probs = F.softmax(fused_ncm_logits, dim=-1)

    if f_type == 'prob':
        test_fused_score = (1.0 - alpha_star) * test_linear_probs + alpha_star * test_ncm_probs
    else:
        test_fused_score = (1.0 - alpha_star) * test_logits + alpha_star * fused_ncm_logits

    test_dpfc_acc, test_dpfc_preds = evaluate_accuracy(test_fused_score, test_labels)

    # L2: 双重 McNemar 统计检验
    corr_lin = (test_linear_preds == test_labels)
    corr_dpfc = (test_dpfc_preds == test_labels)
    corr_ncm = (test_ncm_preds == test_labels)

    # 检验 1: DPFC vs Linear
    b_dpfc = int(((corr_lin == True) & (corr_dpfc == False)).sum().item())
    c_dpfc = int(((corr_lin == False) & (corr_dpfc == True)).sum().item())
    a_dpfc = int(((corr_lin == True) & (corr_dpfc == True)).sum().item())
    d_dpfc = int(((corr_lin == False) & (corr_dpfc == False)).sum().item())
    stat_dpfc, p_dpfc, method_dpfc = mcnemar_test(b_dpfc, c_dpfc)

    # 检验 2: NCM-only vs Linear
    b_ncm = int(((corr_lin == True) & (corr_ncm == False)).sum().item())
    c_ncm = int(((corr_lin == False) & (corr_ncm == True)).sum().item())
    a_ncm = int(((corr_lin == True) & (corr_ncm == True)).sum().item())
    d_ncm = int(((corr_lin == False) & (corr_ncm == False)).sum().item())
    stat_ncm, p_ncm, method_ncm = mcnemar_test(b_ncm, c_ncm)

    # L3: 产物矩阵深度落盘 (同时支持 dpfc_preds.npz 命名与标准输出目录)
    output_dir = os.path.abspath(args.output_dir) if hasattr(args, "output_dir") and args.output_dir else os.path.join(PROJECT_ROOT, "save", "diag")
    os.makedirs(output_dir, exist_ok=True)
    report_json_path = os.path.join(output_dir, "dpfc_eval_results.json")
    preds_npz_path = os.path.join(output_dir, "dpfc_preds.npz")

    np.savez_compressed(
        preds_npz_path,
        true_labels=test_labels.numpy(),
        linear_probs=test_linear_probs.numpy(),
        ncm_sim=test_ncm_sim.numpy(),
        preds_linear=test_linear_preds.numpy(),
        preds_ncm=test_ncm_preds.numpy(),
        preds_dpfc=test_dpfc_preds.numpy()
    )

    delta_acc = test_dpfc_acc - test_linear_acc

    # N6: 补全 Dev 侧基准指标记录
    results_dict = {
        "checkpoint": args.model_path,
        "prototypes_initialized": f"{num_init}/{total_protos}",
        "dev_baselines": {
            "dev_linear_acc": round(dev_linear_acc, 2),
            "dev_ncm_acc": round(dev_ncm_acc, 2),
            "dev_dpfc_acc": round(dev_dpfc_acc, 2)
        },
        "linear_only": {"test_acc": round(test_linear_acc, 2)},
        "ncm_only": {
            "best_tau": best_ncm_tau,
            "test_acc": round(test_ncm_acc, 2),
            "mcnemar_vs_linear": {
                "contingency_table": {"a_both_correct": a_ncm, "b_lin_only": b_ncm, "c_ncm_only": c_ncm, "d_both_wrong": d_ncm},
                "stat": round(stat_ncm, 4),
                "p_value": float(f"{p_ncm:.6e}"),
                "method": method_ncm,
                "is_significant_positive": bool(p_ncm < 0.05 and c_ncm > b_ncm),
                "is_significant_negative": bool(p_ncm < 0.05 and b_ncm > c_ncm)
            }
        },
        "dpfc_fused": {
            "mode": f_type,
            "best_tau": tau_star,
            "best_alpha": alpha_star,
            "test_acc": round(test_dpfc_acc, 2),
            "delta_acc": round(delta_acc, 2),
            "mcnemar_vs_linear": {
                "contingency_table": {"a_both_correct": a_dpfc, "b_lin_only": b_dpfc, "c_dpfc_only": c_dpfc, "d_both_wrong": d_dpfc},
                "stat": round(stat_dpfc, 4),
                "p_value": float(f"{p_dpfc:.6e}"),
                "method": method_dpfc,
                "is_significant_positive": bool(p_dpfc < 0.05 and c_dpfc > b_dpfc),
                "is_significant_negative": bool(p_dpfc < 0.05 and b_dpfc > c_dpfc)
            }
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("                原型与后验融合测试集最终评测报告")
    print("=" * 70)
    print(f" 1. 纯 Linear 分类头准确率:   {test_linear_acc:.2f}% (Dev: {dev_linear_acc:.2f}%)")
    print(f" 2. 纯 NCM 原型分类准确率:    {test_ncm_acc:.2f}% (Dev: {dev_ncm_acc:.2f}%, 专属最优 tau={best_ncm_tau})")
    print(f" 3. DPFC 融合分类头准确率:    {test_dpfc_acc:.2f}% (Dev: {dev_dpfc_acc:.2f}%, alpha={alpha_star}, tau={tau_star}, mode={f_type})")
    print(f" 4. 绝对准确率改善幅度 (Δ):    {delta_acc:+.2f}%")
    print("-" * 70)
    print(f" [DPFC vs Linear 配对检验 ({method_dpfc})]:")
    print(f"    - 列联表 (b={b_dpfc}, c={c_dpfc}) | stat = {stat_dpfc:.4f} | p-value = {p_dpfc:.6e}")
    if p_dpfc < 0.05 and c_dpfc > b_dpfc:
        print("    --> [显著提升] DPFC 显著性正向提升通过 (p < 0.05 且 c > b)！")
    elif p_dpfc < 0.05 and b_dpfc > c_dpfc:
        print("    --> [显著劣化] DPFC 显著性负向劣化 (p < 0.05 且 b > c)！")
    else:
        print("    --> [不显著] DPFC 显著性检验未通过 (p >= 0.05)，处于噪声带宽内。")

    print("-" * 70)
    print(f" [NCM-only vs Linear 配对检验 ({method_ncm})]:")
    print(f"    - 列联表 (b={b_ncm}, c={c_ncm}) | stat = {stat_ncm:.4f} | p-value = {p_ncm:.6e}")

    # 零传输摩擦: 紧凑 JSON 单行打印
    print("\n" + "=" * 75)
    print(" [EVAL_NCM_COMPACT_JSON]")
    print(json.dumps(results_dict, ensure_ascii=False))
    print("=" * 75 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CASE-EPCL 原型最近邻分类 (NCM) 评测与特征落盘流水线")
    parser.add_argument("--model_path", type=str, default="save/v9_trial10/CASE_7499_37.3144_composite_best", help="目标检查点路径")
    parser.add_argument("--batch_size", type=int, default=32, help="评估 Batch 大小")
    parser.add_argument("--output_dir", type=str, default="save/diag", help="评测结果与特征 NPZ 保存目录")
    args = parser.parse_args()

    run_ncm_evaluation(args)
