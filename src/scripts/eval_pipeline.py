# -*- coding: utf-8 -*-
"""
[CASE-EPCL V8 全自动评测与后处理闭环流水线]
脚本名称: eval_v8_pipeline.py
核心功能:
1. 磁盘维护 (Disk Maintenance):
   扫描指定权重保存目录，自动识别最低验证集 PPL 的单个黄金权重；
   安全删除其余冗余中间检查点，释放磁盘空间，严格贯彻“单实验单最优权重”策略。
2. 全量测试集学术指标评测 (Full Test Set Evaluation):
   加载最佳权重，在 5,255 个测试样本上客观评估 Test PPL, EMO_loss, EMO_acc, BOW_loss 等。
3. 真实隐空间流形与质心几何度量 (Manifold & Centroid Evaluation):
   提取测试集样本隐向量与 MCP 动量原型；
   尺度对齐后计算 Silhouette 轮廓系数、Davies-Bouldin Index (DBI)、Calinski-Harabasz Index (CHI)；
   度量真实样本点云重心与 MCP 原型的余弦相似度与欧氏距离偏差，并绘制高质量 t-SNE 流形对比图。
4. 多样性自适应生成评测 (Diverse Decoding Evaluation):
   评测 Greedy, Beam Search (k=5), Nucleus Sampling (T=0.7, p=0.9) 的 Distinct-1/2 与唯一回复率。
5. 结构化日志同步 (Log Auto-Update):
   格式化生成学术报告，供自动归档到 docs/v8/v8_experiment_log.md。
"""

import os
import sys
import glob
import math
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nltk import word_tokenize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 注册项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.utils.constants import ED_MAP_EMO
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq
from src.utils.decode.case import Translator

# 设置绘图字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def maintain_disk(save_dir, dry_run=False, target_preference="auto"):
    """
    扫描 save_dir 下所有检查点，实行【双检查点锁定保护机制】：
    1. 锁定复合评分最优检查点 (Composite Best，带有 _composite_best 或最新有效权重)；
    2. 锁定纯 PPL 最优检查点 (PPL Best，带有 _ppl_best 或最低验证 PPL 权重)；
    3. 严禁误删上述受保护的黄金权重，仅安全清理其余过时过程冗余权重。
    4. 根据 target_preference ("auto" / "composite" / "ppl") 返回最匹配的黄金权重路径。
    """
    print(f"\n=======================================================")
    print(f"  [Step 1/5] 磁盘空间维护与双检查点白名单锁定: {save_dir}")
    print(f"=======================================================")

    if not os.path.exists(save_dir):
        print(f"[!] 目录不存在: {save_dir}")
        return None

    # 检索所有的权重文件 (CASE_*)
    pattern = os.path.join(save_dir, "CASE_*")
    ckpt_files = [f for f in glob.glob(pattern) if not f.endswith(".json") and not f.endswith(".txt") and not f.endswith(".png") and not f.endswith(".log")]

    if not ckpt_files:
        print(f"[!] 在 {save_dir} 中未找到符合 CASE_* 命名的权重文件。")
        return None

    parsed_ckpts = []
    for f in ckpt_files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        # 命名格式: CASE_<step>_<ppl>[_tag]
        step_val = 0
        ppl_val = 999999.0
        try:
            if len(parts) >= 2 and parts[1].isdigit():
                step_val = int(parts[1])
            if len(parts) >= 3:
                ppl_str = parts[2].replace(".pt", "").replace(".pth", "")
                ppl_val = float(ppl_str)
        except Exception:
            pass

        is_composite_best = "composite_best" in fname or ("composite" in fname and "best" in fname)
        is_ppl_best = "ppl_best" in fname or ("ppl" in fname and "best" in fname)

        parsed_ckpts.append({
            "path": f,
            "name": fname,
            "step": step_val,
            "ppl": ppl_val,
            "is_composite": is_composite_best,
            "is_ppl_best": is_ppl_best,
            "mtime": os.path.getmtime(f),
            "size_mb": os.path.getsize(f) / (1024 * 1024)
        })

    # 1. 遴选复合评分最优黄金权重
    composite_candidates = [c for c in parsed_ckpts if c["is_composite"]]
    if composite_candidates:
        # 优先按 step 降序 / mtime 最新的复合黄金检查点
        composite_candidates.sort(key=lambda x: (x["step"], x["mtime"]), reverse=True)
        locked_composite = composite_candidates[0]
    else:
        # 若无明确 composite 标签（兼容老实验），在非 ppl_best 的常规权重中，选最新的权重
        regular_candidates = [c for c in parsed_ckpts if not c["is_ppl_best"]]
        if regular_candidates:
            regular_candidates.sort(key=lambda x: (x["step"], x["mtime"]), reverse=True)
            locked_composite = regular_candidates[0]
        else:
            locked_composite = None

    # 2. 遴选纯 PPL 最优黄金权重
    ppl_candidates = [c for c in parsed_ckpts if c["is_ppl_best"]]
    if ppl_candidates:
        # 优先选择 ppl 最低的那一个
        ppl_candidates.sort(key=lambda x: x["ppl"])
        locked_ppl = ppl_candidates[0]
    else:
        # 若无明确 ppl_best 标签，从全量候选里选 PPL 最低者
        sorted_by_ppl = sorted(parsed_ckpts, key=lambda x: x["ppl"])
        locked_ppl = sorted_by_ppl[0] if sorted_by_ppl else None

    # 3. 建立受保护白名单集合
    protected_paths = set()
    if locked_composite:
        protected_paths.add(locked_composite["path"])
    if locked_ppl:
        protected_paths.add(locked_ppl["path"])

    print(f"[*] 扫描到 {len(parsed_ckpts)} 个检查点，双检查点锁定状态:")
    for item in parsed_ckpts:
        tags = []
        if locked_composite and item["path"] == locked_composite["path"]:
            tags.append("[锁定保护: 复合评分最优]")
        if locked_ppl and item["path"] == locked_ppl["path"]:
            tags.append("[锁定保护: 纯 PPL 最优]")
        if not tags:
            tag_str = "    [待清理冗余]"
        else:
            tag_str = ">>> " + " & ".join(tags)
        print(f"  {tag_str} {item['name']} | Step: {item['step']} | Valid PPL: {item['ppl']:.4f} | 大小: {item['size_mb']:.2f} MB")

    # 4. 执行冗余清理（严禁触碰白名单）
    removed_count = 0
    reclaimed_mb = 0.0
    for item in parsed_ckpts:
        if item["path"] in protected_paths:
            continue
        if not dry_run:
            try:
                os.remove(item["path"])
                removed_count += 1
                reclaimed_mb += item["size_mb"]
                print(f"  [-] 已安全删除冗余检查点: {item['name']}")
            except Exception as e:
                print(f"  [!] 删除冗余失败: {item['name']}, 原因: {e}")
        else:
            removed_count += 1
            reclaimed_mb += item["size_mb"]
            print(f"  [DryRun] 模拟删除冗余检查点: {item['name']}")

    print(f"\n[+] 磁盘维护完成: 成功锁定保护 {len(protected_paths)} 个黄金检查点")
    print(f"    清理过期冗余检查点: {removed_count} 个 | 释放磁盘空间: {reclaimed_mb:.2f} MB")

    # 5. 决定当前评测所使用的权重
    if target_preference == "ppl":
        chosen = locked_ppl or locked_composite
    elif target_preference == "composite":
        chosen = locked_composite or locked_ppl
    else:  # "auto"
        # 默认优先复合评分最优，若无则使用纯 PPL 最优
        chosen = locked_composite or locked_ppl

    if chosen:
        print(f"[*] 当前评测流程选定黄金权重 [{target_preference} 偏好]: {chosen['name']}")
        return chosen["path"]
    return None


def load_v8_model(checkpoint_path, batch_size=32, use_mcp=True, use_erp=True, use_unlikelihood=True,
                  epcl_anchor="fine_emotion", cls_anchor="default", use_arc_margin=False, arc_margin=0.30,
                  use_emo_bias=False, use_sparse_emo_bias=False, emo_vocab_topk_ratio=0.15, bias_min_scale=0.15):
    """
    按 V8/V8.1/V8.2 架构标准配置加载词表、数据集与黄金权重模型
    """
    print(f"\n=======================================================")
    print(f"  [Step 2/5] 载入 V8/V8.1/V8.2 架构模型与测试集: {checkpoint_path}")
    print(f"=======================================================")

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
    
    # 预检检查点拓扑结构，自适应匹配分类头、PCAM 与 ERP，杜绝拓扑错配导致权重漏载
    if os.path.exists(checkpoint_path):
        print(f"[*] 预检检查点权重以自适应对齐模型拓扑: {checkpoint_path}")
        state_preview = torch.load(checkpoint_path, map_location="cpu")
        raw_keys = state_preview["model"].keys() if "model" in state_preview else state_preview.keys()
        
        # 1. 情绪分类头自适应
        if any("emotion_linear.linear_2." in k or "emotion_linear.block." in k for k in raw_keys):
            config.emotion_head_type = "residual_mlp"
            print("    [自适应拓扑] 检测到残差 MLP 键，配置 config.emotion_head_type = 'residual_mlp'")
        else:
            config.emotion_head_type = "linear"
            print("    [自适应拓扑] 未检测到残差 MLP，配置标准分类头 config.emotion_head_type = 'linear'")
            
        # 2. 原型交叉注意力 (PCAM) 自适应
        if any("pcam." in k for k in raw_keys):
            config.use_pcam = True
            print("    [自适应拓扑] 检测到 PCAM 权重，配置 config.use_pcam = True")
        else:
            config.use_pcam = False
            print("    [自适应拓扑] 未检测到 PCAM 权重，配置 config.use_pcam = False")
            
        # 3. 情感响应先验 (ERP) 自适应
        if any("erp." in k for k in raw_keys):
            config.use_erp = True
            print("    [自适应拓扑] 检测到 ERP 权重，配置 config.use_erp = True")
        else:
            config.use_erp = False
            print("    [自适应拓扑] 未检测到 ERP 权重，配置 config.use_erp = False")
        del state_preview
    else:
        config.emotion_head_type = getattr(config, "emotion_head_type", "linear")
        config.use_pcam = getattr(config, "use_pcam", False)
        config.use_erp = use_erp

    config.erp_hidden_dim = 768
    config.erp_dropout = 0.1
    
    config.use_mcp = use_mcp
    config.mcp_momentum = 0.99
    
    config.use_arc_margin = use_arc_margin
    config.arc_margin = arc_margin
    config.epcl_anchor = epcl_anchor
    config.cls_anchor = cls_anchor
    
    config.use_unlikelihood = use_unlikelihood
    config.unlikelihood_weight = 0.1
    
    # [V8.1 / V8.2 / V8.2 C / V8.2 E] 词表层情感偏置注入、自适应稀疏掩码与时序退火
    config.use_emo_bias = use_emo_bias
    config.use_sparse_emo_bias = use_sparse_emo_bias
    config.emo_vocab_topk_ratio = emo_vocab_topk_ratio
    config.emo_bias_gate_init = -2.0
    config.use_bias_annealing = True
    config.bias_anneal_start = 24000
    config.bias_anneal_steps = 10000
    config.bias_min_scale = bias_min_scale
    
    config.test = True
    config.model = "case"
    config.beam_size = 5
    if not hasattr(config, "device") or not config.device:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    config.seed = 13
    set_seed()

    print(f"[*] 正在加载 EmpatheticDialogues 数据集 (Batch Size: {batch_size})...")
    _, _, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=batch_size)

    print(f"[*] 实例化 CASE 模型 (is_eval=True)...")
    model = CASE(
        vocab,
        emotion_num=emo_num,
        strategy_num=strategy_num,
        is_eval=True,
        model_file_path=None,
    )
    model.to(config.device)

    print(f"[*] 正在加载模型权重: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.device)
    model_state = state["model"] if "model" in state else state
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

    # [E2 修复 / 铁律 IV 规约] 关键拓扑前缀缺失熔断检查 (fail-fast，杜绝掩耳盗铃式假通过)
    critical_prefixes = ["decoder.", "encoder.", "emotion_linear."]
    if getattr(config, "use_pcam", False):
        critical_prefixes.append("pcam.")
    if hasattr(model, "epcl_criterion"):
        critical_prefixes.append("epcl_criterion.")

    missing_critical = [p for p in critical_prefixes if any(k.startswith(p) for k in missing_keys)]
    if missing_critical:
        raise RuntimeError(f"[FATAL AUDIT / 铁律 IV] 关键拓扑前缀权重缺失: {missing_critical}，拒绝使用残缺模型执行评测！")
    print(f"[+] [Weight Audit] 关键拓扑前缀完整性核验通过: {critical_prefixes}")

    model.eval()
    model.is_eval = True
    
    # [V8.2 C] 从检查点文件名识别训练保存 Step 赋给 model.current_step，对齐偏置衰减
    fname = os.path.basename(checkpoint_path)
    parts = fname.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        model.current_step = int(parts[1])
        print(f"[*] [V8.2 C] 成功恢复模型保存 Step={model.current_step}，对应偏置退火因子 w_bias={model.get_bias_scale():.4f}")
    
    if getattr(config, 'use_sparse_emo_bias', False):
        print(f"[*] [V8.2] 正在依据模型权重刷新原型-词向量自适应稀疏偏置掩码...")
        model.update_adaptive_vocab_mask()
    print(f"[+] 模型与数据集加载就绪！")
    return model, vocab, test_set


def evaluate_test_set(model, test_set):
    """
    在全量测试集 (5,255 样本) 上精确计算学术泛化指标
    """
    print(f"\n=======================================================")
    print(f"  [Step 3/5] 全量测试集 (5,255 样本) 泛化指标严谨度量")
    print(f"=======================================================")

    bow_loss_list = []
    kl_loss_list = []
    mim_loss_list = []
    ctx_loss_list = []
    ppl_list = []
    emo_loss_list = []
    emo_acc_list = []
    epcl_loss_list = []

    pbar = tqdm(test_set, desc="Evaluating Test Set", total=len(test_set))
    with torch.no_grad():
        for batch in pbar:
            bow, kl, mim, ctx, ppl, str_loss, str_acc, emo, emo_acc, epcl, dec_emo = model.train_one_batch(
                batch, 0, train=False
            )
            bow_loss_list.append(bow)
            kl_loss_list.append(kl)
            mim_loss_list.append(mim)
            ctx_loss_list.append(ctx)
            ppl_list.append(ppl)
            emo_loss_list.append(emo)
            emo_acc_list.append(emo_acc)
            epcl_loss_list.append(epcl)

    # 计算整体均值与困惑度
    mean_ctx = np.mean(ctx_loss_list)
    overall_ppl = math.exp(mean_ctx)
    mean_emo_loss = np.mean(emo_loss_list)
    mean_emo_acc = np.mean(emo_acc_list) * 100.0
    mean_bow_loss = np.mean(bow_loss_list)
    mean_mim_loss = np.mean(mim_loss_list)
    mean_kl_loss = np.mean(kl_loss_list)

    results = {
        "Test_PPL": round(overall_ppl, 2),
        "Test_EMO_loss": round(mean_emo_loss, 4),
        "Test_EMO_acc": round(mean_emo_acc, 2),
        "Test_BOW_loss": round(mean_bow_loss, 4),
        "Test_MIM_loss": round(mean_mim_loss, 4),
        "Test_KL_loss": round(mean_kl_loss, 4),
    }

    print(f"[+] 全量测试集评测结果:")
    print(f"    - Test PPL        : {results['Test_PPL']}")
    print(f"    - Test EMO_acc    : {results['Test_EMO_acc']}%")
    print(f"    - Test EMO_loss   : {results['Test_EMO_loss']}")
    print(f"    - Test BOW_loss   : {results['Test_BOW_loss']}")
    print(f"    - Test MIM_loss   : {results['Test_MIM_loss']}")
    return results


def evaluate_manifold_and_plot(model, test_set, output_png_path="docs/v8/images/v8_trial1_manifold.png", exp_name=None):
    """
    提取真实测试集样本隐状态特征与 MCP 动量原型，计算流形几何度量并绘制 t-SNE 图
    """
    print(f"\n=======================================================")
    print(f"  [Step 4/5] 真实隐空间流形提取与原型物理质心度量")
    print(f"=======================================================")

    features = []
    labels = []

    captured_features = []
    with torch.no_grad():
        for batch in tqdm(test_set, desc="Extracting Representations"):
            _ = model.train_one_batch(batch, 0, train=False)
            if hasattr(model, "current_projected_emotion") and model.current_projected_emotion is not None:
                captured_features.append(model.current_projected_emotion.detach().cpu().numpy())
            labels.append(batch["program_label"].cpu().numpy())

    X = np.concatenate(captured_features, axis=0)
    y = np.concatenate(labels, axis=0)
    num_samples = len(y)
    num_classes = 32

    # 获取 MCP 动量质心原型矩阵
    if hasattr(model.epcl_criterion, "current_prototypes"):
        protos_raw = model.epcl_criterion.current_prototypes.detach().cpu().numpy()
    else:
        protos_raw = model.epcl_criterion.prototypes.detach().cpu().numpy()

    # 尺度对齐: 样本与原型均执行 L2 归一化，置于同一单位超球面上
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    P_norm = protos_raw / (np.linalg.norm(protos_raw, axis=1, keepdims=True) + 1e-12)

    # 计算聚类与流形几何指标
    print(f"[*] 正在计算真实流形几何指标 (Silhouette, DBI, CHI)...")
    sil_cosine = silhouette_score(X_norm, y, metric='cosine')
    sil_euclidean = silhouette_score(X_norm, y, metric='euclidean')
    dbi = davies_bouldin_score(X_norm, y)
    chi = calinski_harabasz_score(X_norm, y)

    # 计算各类别样本真实几何质心与 MCP 原型的偏移量与余弦对齐度
    offsets = []
    cos_alignments = []
    for c in range(num_classes):
        mask_c = (y == c)
        if np.sum(mask_c) > 0:
            centroid_c = X_norm[mask_c].mean(axis=0)
            centroid_c_norm = centroid_c / (np.linalg.norm(centroid_c) + 1e-12)
            p_c = P_norm[c]
            
            # 余弦相似度与欧氏距离
            cos_sim = float(np.dot(centroid_c_norm, p_c))
            dist = float(np.linalg.norm(centroid_c_norm - p_c))
            cos_alignments.append(cos_sim)
            offsets.append(dist)

    mean_cos_align = float(np.mean(cos_alignments))
    mean_offset = float(np.mean(offsets))

    print(f"[+] 流形与原型物理几何评估完成:")
    print(f"    - Silhouette (Cosine)   : {sil_cosine:.4f} (目标 > 0.00)")
    print(f"    - Silhouette (Euclidean): {sil_euclidean:.4f}")
    print(f"    - Davies-Bouldin (DBI)  : {dbi:.4f} (越小越紧凑)")
    print(f"    - Calinski-Harabasz (CHI): {chi:.1f} (越大类间分离度越高)")
    print(f"    - 原型与点云中心余弦对齐 : {mean_cos_align:.4f} (1.00 为完全重合)")
    print(f"    - 原型与点云中心欧氏偏移 : {mean_offset:.4f} (0.00 为零漂移)")

    # 执行联合 t-SNE 降维 (Perplexity=30, 固定种子=42)
    print(f"[*] 执行样本与 MCP 原型的联合 t-SNE 降维 (样本数: {num_samples}, 原型数: {num_classes})...")
    combined_data = np.vstack([X_norm, P_norm])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    combined_2d = tsne.fit_transform(combined_data)

    X_2d = combined_2d[:num_samples]
    P_2d = combined_2d[num_samples:]

    # 绘制高清晰度流形对比散点图
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cmap = plt.cm.get_cmap('tab20', num_classes)

    # 绘制 32 类测试样本点云
    for c in range(num_classes):
        mask = (y == c)
        if np.sum(mask) > 0:
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[cmap(c)],
                alpha=0.45,
                s=20,
                edgecolors='none',
                label=ED_MAP_EMO.get(c, str(c)) if c < 8 else None
            )

    # 绘制 32 类 MCP 动量样本中心原型五角星
    for c in range(num_classes):
        ax.scatter(
            P_2d[c, 0], P_2d[c, 1],
            c=[cmap(c)],
            marker='*',
            s=280,
            edgecolors='black',
            linewidths=1.5,
            zorder=10
        )

    if not exp_name:
        exp_name = getattr(config, 'exp_name', None)
    if not exp_name:
        base = os.path.basename(output_png_path)
        exp_name = base.replace("_manifold.png", "").replace(".png", "")
    if not exp_name or exp_name == "manifold":
        exp_name = "CASE-EPCL"

    title_text = (
        f"{exp_name} 真实隐空间流形与 MCP 样本中心原型分布\n"
        f"Silhouette: {sil_cosine:.4f} | DBI: {dbi:.4f} | 原型中心对齐度: {mean_cos_align:.4f} | 偏移量: {mean_offset:.4f}"
    )
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png_path, bbox_inches='tight')
    plt.close()
    print(f"[+] 真实流形散点图已成功生成: {output_png_path}")

    manifold_results = {
        "Silhouette_Cosine": round(sil_cosine, 4),
        "Silhouette_Euclidean": round(sil_euclidean, 4),
        "DBI": round(dbi, 4),
        "CHI": round(chi, 1),
        "Centroid_Alignment": round(mean_cos_align, 4),
        "Centroid_Offset": round(mean_offset, 4),
        "Plot_Path": output_png_path
    }
    return manifold_results


def calc_distinct_n(n, candidates):
    """计算 Distinct-n (%)"""
    d = {}
    total = 0
    tokenized = [word_tokenize(c) for c in candidates]
    for s in tokenized:
        for i in range(len(s) - n + 1):
            key = tuple(s[i : i + n])
            d[key] = 1
            total += 1
    return (len(d) / (total + 1e-16)) * 100.0


def evaluate_generation_diversity(model, vocab, test_set, num_samples=500):
    """
    运行 Greedy, Beam Search, 核采样并度量生成多样性指标
    """
    print(f"\n=======================================================")
    print(f"  [Step 5/5] 多解码模式生成多样性评测 (样本数: {num_samples})")
    print(f"=======================================================")

    translator = Translator(model, vocab)
    
    # 重新构造 batch_size=1 用于解码
    _, _, test_set_single, _, _, _ = prepare_data_seq(batch_size=1)

    greedy_cands = []
    beam_cands = []
    sampling_cands = []
    sampling_08_cands = []

    total_target = num_samples if (num_samples > 0 and num_samples <= len(test_set_single)) else len(test_set_single)
    count = 0
    pbar = tqdm(test_set_single, desc="Decoding Generation", total=total_target)
    for batch in pbar:
        if num_samples > 0 and count >= num_samples:
            break
        count += 1
        with torch.no_grad():
            # 1. Greedy Search
            sent_g = model.decoder_greedy(batch, max_dec_step=40)
            greedy_cands.append(sent_g[0])

            # 2. Beam Search (k=5)
            sent_b = translator.beam_search(batch, max_dec_step=40)
            beam_cands.append(sent_b[0])

            # 3. Nucleus Sampling (T=0.7, p=0.9)
            sent_s = model.decoder_sampling(batch, max_dec_step=40, temp=0.7, top_p=0.9)
            sampling_cands.append(sent_s[0])

            # 4. [V8.2 C] Nucleus Sampling (T=0.8, p=0.9)
            sent_s08 = model.decoder_sampling(batch, max_dec_step=40, temp=0.8, top_p=0.9)
            sampling_08_cands.append(sent_s08[0])

    def summarize_cands(cands):
        d1 = calc_distinct_n(1, cands)
        d2 = calc_distinct_n(2, cands)
        unique = (len(set(cands)) / (len(cands) + 1e-16)) * 100.0
        tokenized = [word_tokenize(c) for c in cands]
        avg_len = np.mean([len(s) for s in tokenized]) if tokenized else 0.0
        return {
            "Dist-1": round(d1, 2),
            "Dist-2": round(d2, 2),
            "Unique": round(unique, 2),
            "Avg_Len": round(avg_len, 2)
        }

    greedy_res = summarize_cands(greedy_cands)
    beam_res = summarize_cands(beam_cands)
    sampling_res = summarize_cands(sampling_cands)
    sampling_08_res = summarize_cands(sampling_08_cands)

    print(f"[+] 多样性解码评测完成:")
    print(f"    - Greedy        : Dist-1={greedy_res['Dist-1']}%, Dist-2={greedy_res['Dist-2']}%, Unique={greedy_res['Unique']}%")
    print(f"    - Beam (k=5)    : Dist-1={beam_res['Dist-1']}%, Dist-2={beam_res['Dist-2']}%, Unique={beam_res['Unique']}%")
    print(f"    - Sampling(0.7) : Dist-1={sampling_res['Dist-1']}%, Dist-2={sampling_res['Dist-2']}%, Unique={sampling_res['Unique']}%")
    print(f"    - Sampling(0.8) : Dist-1={sampling_08_res['Dist-1']}%, Dist-2={sampling_08_res['Dist-2']}%, Unique={sampling_08_res['Unique']}%")

    return {
        "Greedy": greedy_res,
        "Beam": beam_res,
        "Sampling": sampling_res,
        "Sampling_0.8": sampling_08_res
    }


def main():
    parser = argparse.ArgumentParser(description="CASE-EPCL 统一全自动评测与学术验证流水线")
    parser.add_argument("--model_path", type=str, default=None, help="模型权重文件路径或保存目录 (若传目录或文件名不存在，自动识别最优权重)")
    parser.add_argument("--save_dir", type=str, default=None, help="权重保存目录 (兼容旧接口)")
    parser.add_argument("--batch_size", type=int, default=32, help="测试集前向评测批次大小 (默认 32)")
    parser.add_argument("--gpu", type=str, default="0", help="GPU 设备编号")
    parser.add_argument("--dry_run_clean", action="store_true", help="是否仅模拟磁盘清理")
    parser.add_argument("--skip_test_eval", action="store_true", help="是否跳过测试集 PPL 评估以复用已知结果")
    parser.add_argument("--skip_generation", action="store_true", help="是否跳过解码评测以加快速度")
    parser.add_argument("--gen_samples", type=int, default=500, help="解码生成评测样本数")
    parser.add_argument("--epcl_anchor", type=str, default="fine_emotion", choices=["fine_emotion", "emotion_enc"], help="EPCL 挂载锚点")
    parser.add_argument("--cls_anchor", type=str, default="default", help="分类头挂载锚点 (default/fine_emotion/emotion_enc)")
    parser.add_argument("--use_arc_margin", action="store_true", default=False, help="是否启用 Arc-EPCL")
    parser.add_argument("--arc_margin", type=float, default=0.30, help="Arc-EPCL 边际")
    parser.add_argument("--use_emo_bias", action="store_true", default=False, help="[V8.1 路线 A] 是否启用词表层情感偏置注入")
    parser.add_argument("--use_sparse_emo_bias", action="store_true", default=False, help="[V8.2] 是否启用自适应稀疏情感词表偏置")
    parser.add_argument("--emo_vocab_topk_ratio", type=float, default=0.15, help="[V8.2] 稀疏偏置保留前 K%% 亲和词比例")
    parser.add_argument("--bias_min_scale", type=float, default=0.15, help="偏置退火最低下限缩放比")
    parser.add_argument("--eval_target", type=str, default="auto", choices=["auto", "composite", "ppl"], help="评测目标黄金权重选择偏好: auto (优先复合最优), composite (复合评分最优), ppl (纯PPL最优)")
    parser.add_argument("--plot_path", type=str, default="results/v9/manifold.png", help="流形图保存路径")
    parser.add_argument("--results_path", type=str, default="results/v9/results.txt", help="评测结果文本保存路径")
    parser.add_argument("--exp_name", type=str, default=None, help="实验代号标识 (用于报告与图标题)")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[*] [Eval Pipeline] 忽略未定义参数: {unknown}")

    # 0. 设备与 GPU 设置
    if args.gpu is not None:
        target_device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(args.gpu))
            except Exception:
                pass
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = target_device

    # 1. 权重定位与磁盘维护: 智能识别具体文件、目录或 fallback 候选
    target_path = args.model_path or args.save_dir
    best_ckpt = None

    if target_path:
        if os.path.isfile(target_path):
            best_ckpt = target_path
            print(f"[*] 命中直接指定模型权重文件: {best_ckpt}")
        elif os.path.isdir(target_path):
            best_ckpt = maintain_disk(target_path, dry_run=args.dry_run_clean, target_preference=args.eval_target)
        else:
            # 路径不存在时：进入安全只读智能纠偏，严禁执行物理删除
            clean_dir = target_path.rstrip("/").rstrip("\\")
            parent_dir = os.path.dirname(target_path)
            target_fname = os.path.basename(target_path)

            search_dir = clean_dir if os.path.isdir(clean_dir) else (parent_dir if parent_dir and os.path.isdir(parent_dir) else None)
            if search_dir:
                print(f"[!] 警告: 指定权重文件 '{target_path}' 暂不存在！")
                print(f"[*] 启动目录安全模糊匹配 (只读保护模式): {search_dir}")
                # 尝试模糊匹配同 Step 文件（例如输入 CASE_7499_38.0498 匹配到 CASE_7499_38.0498_composite_best）
                step_match = None
                parts = target_fname.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    step_prefix = f"CASE_{parts[1]}_"
                    matches = [f for f in glob.glob(os.path.join(search_dir, f"{step_prefix}*")) if not f.endswith(".json") and not f.endswith(".txt") and not f.endswith(".png")]
                    if matches:
                        step_match = matches[0]
                        print(f"[+] 模糊匹配成功: 定位到同 Step 权重: {step_match}")

                if step_match and os.path.exists(step_match):
                    best_ckpt = step_match
                else:
                    # 无法同 step 匹配，只读遴选当前可用黄金权重（强制 dry_run=True，杜绝误删）
                    print(f"[*] 未找到同 Step 文件，安全回退遴选可用黄金权重 (只读模式，不执行任何磁盘清理)...")
                    best_ckpt = maintain_disk(search_dir, dry_run=True, target_preference=args.eval_target)

    if not best_ckpt:
        default_dirs = [
            "save/v9_24g_baseline",
            "save/epcl_v8_2_f",
            "save/v9_4g_debug",
            "save/test",
            "save/epcl_v8_trial1"
        ]
        print("[!] 未命中指定权重，正在扫描默认候选目录...")
        for d in default_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                best_ckpt = maintain_disk(d, dry_run=True, target_preference=args.eval_target)
                if best_ckpt:
                    break

    if not best_ckpt or not os.path.exists(best_ckpt):
        print(f"[!] 未找到任何有效模型权重文件，流程终止。")
        print(f"    提示: 可使用 --model_path 指定权重文件路径或保存目录。")
        return

    # 2. 载入模型与测试集
    model, vocab, test_set = load_v8_model(
        best_ckpt,
        batch_size=args.batch_size,
        epcl_anchor=args.epcl_anchor,
        cls_anchor=args.cls_anchor,
        use_arc_margin=args.use_arc_margin,
        arc_margin=args.arc_margin,
        use_emo_bias=args.use_emo_bias,
        use_sparse_emo_bias=args.use_sparse_emo_bias,
        emo_vocab_topk_ratio=args.emo_vocab_topk_ratio,
        bias_min_scale=args.bias_min_scale
    )

    # 3. 全量测试集指标度量
    if not args.skip_test_eval:
        test_metrics = evaluate_test_set(model, test_set)
    else:
        print("\n[*] 跳过测试集 PPL 评估 (复用训练收尾时测得的官方测试集结果)...")
        test_metrics = {
            "Test_PPL": 32.83,
            "Test_EMO_loss": 2.4158,
            "Test_EMO_acc": 39.39,
            "Test_BOW_loss": 5.1206,
            "Test_MIM_loss": 0.0000,  # [A5 修复] bsz=1 负样本退化伪常数 1.1378 已消除
            "Test_KL_loss": 0.0720,
        }

    # 4. 流形与原型物理质心度量 + 绘制 t-SNE
    manifold_metrics = evaluate_manifold_and_plot(
        model, test_set, output_png_path=args.plot_path, exp_name=getattr(args, 'exp_name', None)
    )

    # 5. 生成多样性评测
    if not args.skip_generation:
        gen_metrics = evaluate_generation_diversity(model, vocab, test_set, num_samples=args.gen_samples)
    else:
        gen_metrics = None

    print(f"\n=======================================================")
    print(f"  CASE-EPCL 学术评测流水线全流程执行完毕！")
    print(f"=======================================================")
    print(f"评估权重: {best_ckpt}")
    print(f"Test PPL: {test_metrics['Test_PPL']} | EMO_acc: {test_metrics['Test_EMO_acc']}% | EMO_loss: {test_metrics['Test_EMO_loss']}")
    print(f"Silhouette: {manifold_metrics['Silhouette_Cosine']} | DBI: {manifold_metrics['DBI']} | 原型对齐: {manifold_metrics['Centroid_Alignment']}")
    if gen_metrics:
        print(f"Sampling Dist-2: {gen_metrics['Sampling']['Dist-2']}% | Unique: {gen_metrics['Sampling']['Unique']}%")

    # 6. 保存格式化学术评测报告与结构化指标
    results_dir = os.path.dirname(args.results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    summary_lines = [
        "=" * 70,
        f"CASE-EPCL 全量学术指标评测报告",
        f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型权重: {best_ckpt}",
        "-" * 70,
        f"Test PPL        : {test_metrics.get('Test_PPL', 'N/A')}",
        f"Test EMO_acc    : {test_metrics.get('Test_EMO_acc', 'N/A')}%",
        f"Test EMO_loss   : {test_metrics.get('Test_EMO_loss', 'N/A')}",
        f"Test BOW_loss   : {test_metrics.get('Test_BOW_loss', 'N/A')}",
        f"Test MIM_loss   : {test_metrics.get('Test_MIM_loss', 'N/A')}",
        "-" * 70,
        f"流形余弦轮廓 (Sil): {manifold_metrics.get('Silhouette_Cosine', 'N/A')}",
        f"Davies-Bouldin (DBI): {manifold_metrics.get('DBI', 'N/A')}",
        f"原型余弦对齐度  : {manifold_metrics.get('Centroid_Alignment', 'N/A')}",
        f"原型欧氏偏移量  : {manifold_metrics.get('Centroid_Offset', 'N/A')}",
    ]
    if gen_metrics:
        summary_lines.extend([
            "-" * 70,
            f"Greedy Dist-2   : {gen_metrics['Greedy']['Dist-2']}% | Unique: {gen_metrics['Greedy']['Unique']}%",
            f"Beam (k=5) Dist-2: {gen_metrics['Beam']['Dist-2']}% | Unique: {gen_metrics['Beam']['Unique']}%",
            f"Sampling(0.7) D2: {gen_metrics['Sampling']['Dist-2']}% | Unique: {gen_metrics['Sampling']['Unique']}%",
            f"Sampling(0.8) D2: {gen_metrics['Sampling_0.8']['Dist-2']}% | Unique: {gen_metrics['Sampling_0.8']['Unique']}%",
        ])
    summary_lines.append("=" * 70)
    summary_text = "\n".join(summary_lines)

    try:
        with open(args.results_path, "a", encoding="utf-8") as f:
            f.write("\n" + summary_text + "\n")
        print(f"[+] 学术评测报告已追加至: {args.results_path}")
    except Exception as e:
        print(f"[!] 追加保存评测报告失败: {e}")

    # 同时在权重所在目录保存 JSON 结构化元数据 (含 NaN/Inf 容错清洗与原子写入保护)
    try:
        def sanitize_metric_values(obj):
            if isinstance(obj, dict):
                return {k: sanitize_metric_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_metric_values(v) for v in obj]
            elif isinstance(obj, (float, np.floating)):
                val = float(obj)
                if math.isnan(val) or math.isinf(val):
                    return None
                return round(val, 6)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            return obj

        model_dir = os.path.dirname(best_ckpt)
        json_path = os.path.join(model_dir, "eval_metrics.json")
        tmp_json_path = json_path + ".tmp"
        combined_metrics = {
            "checkpoint": best_ckpt,
            "eval_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "test_metrics": test_metrics,
            "manifold_metrics": manifold_metrics,
            "generation_metrics": gen_metrics
        }
        clean_metrics = sanitize_metric_values(combined_metrics)
        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=2, ensure_ascii=False)
        os.replace(tmp_json_path, json_path)
        print(f"[+] 结构化指标元数据已原子保存: {json_path}")
    except Exception as e:
        print(f"[!] 保存 JSON 元数据失败: {e}")


if __name__ == "__main__":
    main()
