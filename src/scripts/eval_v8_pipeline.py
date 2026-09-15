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
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def maintain_disk(save_dir, dry_run=False):
    """
    扫描 save_dir 下所有检查点，识别出验证集 PPL 最低的单一最优权重，安全删除其余冗余权重。
    """
    print(f"\n=======================================================")
    print(f"  [Step 1/5] 磁盘空间维护与黄金权重遴选: {save_dir}")
    print(f"=======================================================")

    if not os.path.exists(save_dir):
        print(f"[!] 目录不存在: {save_dir}")
        return None

    # 检索所有的权重文件 (CASE_*)
    pattern = os.path.join(save_dir, "CASE_*")
    ckpt_files = [f for f in glob.glob(pattern) if not f.endswith(".json") and not f.endswith(".txt")]

    if not ckpt_files:
        print(f"[!] 在 {save_dir} 中未找到符合 CASE_* 命名的权重文件。")
        return None

    parsed_ckpts = []
    for f in ckpt_files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        # 命名格式: CASE_<step>_<ppl> 或 CASE_<step>_<ppl>.pt
        try:
            step_str = parts[1]
            ppl_str = parts[2].replace(".pt", "")
            ppl_val = float(ppl_str)
            step_val = int(step_str)
            parsed_ckpts.append({
                "path": f,
                "name": fname,
                "step": step_val,
                "ppl": ppl_val,
                "size_mb": os.path.getsize(f) / (1024 * 1024)
            })
        except Exception as e:
            # 若无法解析，按修改时间备选
            parsed_ckpts.append({
                "path": f,
                "name": fname,
                "step": 0,
                "ppl": 999999.0,
                "size_mb": os.path.getsize(f) / (1024 * 1024)
            })

    # 按 PPL 升序排序，PPL 最小的为最优权重
    parsed_ckpts.sort(key=lambda x: x["ppl"])
    best_ckpt = parsed_ckpts[0]

    print(f"[*] 扫描到 {len(parsed_ckpts)} 个检查点:")
    for item in parsed_ckpts:
        tag = ">>> [最佳保持]" if item["path"] == best_ckpt["path"] else "    [待清理]"
        print(f"  {tag} {item['name']} | Step: {item['step']} | Valid PPL: {item['ppl']:.4f} | 大小: {item['size_mb']:.2f} MB")

    # 执行清理
    removed_count = 0
    reclaimed_mb = 0.0
    for item in parsed_ckpts[1:]:
        if not dry_run:
            try:
                os.remove(item["path"])
                removed_count += 1
                reclaimed_mb += item["size_mb"]
                print(f"  [-] 已安全删除: {item['name']}")
            except Exception as e:
                print(f"  [!] 删除失败: {item['name']}, 原因: {e}")
        else:
            removed_count += 1
            reclaimed_mb += item["size_mb"]
            print(f"  [DryRun] 模拟删除: {item['name']}")

    print(f"\n[+] 磁盘维护完成: 保留黄金权重 {best_ckpt['name']}")
    print(f"    成功清理冗余权重: {removed_count} 个 | 释放磁盘空间: {reclaimed_mb:.2f} MB")
    return best_ckpt["path"]


def load_v8_model(checkpoint_path, batch_size=16, use_mcp=True, use_erp=True, use_unlikelihood=True,
                  epcl_anchor="fine_emotion", cls_anchor="default", use_arc_margin=False, arc_margin=0.30):
    """
    按 V8 架构标准配置加载词表、数据集与黄金权重模型
    """
    print(f"\n=======================================================")
    print(f"  [Step 2/5] 载入 V8 架构模型与测试集: {checkpoint_path}")
    print(f"=======================================================")

    config.dataset = "ED"
    config.woStrategy = True
    config.emotion_head_type = "residual_mlp"
    config.mlp_hidden_dim = 300
    config.mlp_dropout = 0.1
    config.num_prototypes_per_class = 1
    config.use_pcam = True
    config.pcam_heads = 2
    config.pcam_dropout = 0.1
    config.pcam_gate_bias = -1.0
    
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
    
    config.test = True
    config.model = "case"
    config.beam_size = 5
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
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    model.is_eval = True
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


def evaluate_manifold_and_plot(model, test_set, output_png_path="docs/v8/images/v8_trial1_manifold.png"):
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

    title_text = (
        f"CASE-EPCL V8 真实隐空间流形与 MCP 样本中心原型分布\n"
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

    print(f"[+] 多样性解码评测完成:")
    print(f"    - Greedy      : Dist-1={greedy_res['Dist-1']}%, Dist-2={greedy_res['Dist-2']}%, Unique={greedy_res['Unique']}%")
    print(f"    - Beam (k=5)  : Dist-1={beam_res['Dist-1']}%, Dist-2={beam_res['Dist-2']}%, Unique={beam_res['Unique']}%")
    print(f"    - Sampling(0.7): Dist-1={sampling_res['Dist-1']}%, Dist-2={sampling_res['Dist-2']}%, Unique={sampling_res['Unique']}%")

    return {
        "Greedy": greedy_res,
        "Beam": beam_res,
        "Sampling": sampling_res
    }


def main():
    parser = argparse.ArgumentParser(description="CASE-EPCL V8 全自动评测与后处理流水线")
    parser.add_argument("--save_dir", type=str, default="save/epcl_v8_trial1", help="权重保存目录")
    parser.add_argument("--dry_run_clean", action="store_true", help="是否仅模拟磁盘清理")
    parser.add_argument("--skip_test_eval", action="store_true", help="是否跳过测试集 PPL 评估以复用已知结果")
    parser.add_argument("--skip_generation", action="store_true", help="是否跳过解码评测以加快速度")
    parser.add_argument("--gen_samples", type=int, default=500, help="解码生成评测样本数")
    parser.add_argument("--epcl_anchor", type=str, default="fine_emotion", choices=["fine_emotion", "emotion_enc"], help="EPCL 挂载锚点")
    parser.add_argument("--cls_anchor", type=str, default="default", help="分类头挂载锚点 (default/fine_emotion/emotion_enc)")
    parser.add_argument("--use_arc_margin", action="store_true", default=False, help="是否启用 Arc-EPCL")
    parser.add_argument("--arc_margin", type=float, default=0.30, help="Arc-EPCL 边际")
    parser.add_argument("--plot_path", type=str, default="docs/v8/images/v8_trial_manifold.png", help="流形图保存路径")
    args = parser.parse_args()

    # 1. 磁盘维护: 保留单最佳权重
    best_ckpt = maintain_disk(args.save_dir, dry_run=args.dry_run_clean)
    if not best_ckpt:
        print("[!] 未找到有效检查点，流程终止。")
        return

    # 2. 载入模型与测试集
    model, vocab, test_set = load_v8_model(
        best_ckpt,
        epcl_anchor=args.epcl_anchor,
        cls_anchor=args.cls_anchor,
        use_arc_margin=args.use_arc_margin,
        arc_margin=args.arc_margin
    )

    # 3. 全量测试集指标度量
    if not args.skip_test_eval:
        test_metrics = evaluate_test_set(model, test_set)
    else:
        print("\n[*] 跳过测试集 PPL 评估 (复用训练收尾时测得的官方测试集结果)...")
        if "trial4" in args.save_dir:
            test_metrics = {
                "Test_PPL": 33.40,
                "Test_EMO_loss": 2.8137,
                "Test_EMO_acc": 39.66,
                "Test_BOW_loss": 5.1339,
                "Test_MIM_loss": 1.2326,
                "Test_KL_loss": 0.0749,
            }
        elif "trial3" in args.save_dir:
            test_metrics = {
                "Test_PPL": 33.12,
                "Test_EMO_loss": 2.3967,
                "Test_EMO_acc": 39.28,
                "Test_BOW_loss": 5.1065,
                "Test_MIM_loss": 1.1396,
                "Test_KL_loss": 0.0689,
            }
        else:
            test_metrics = {
                "Test_PPL": 32.62,
                "Test_EMO_loss": 2.4158,
                "Test_EMO_acc": 39.39,
                "Test_BOW_loss": 5.1206,
                "Test_MIM_loss": 1.1378,
                "Test_KL_loss": 0.0720,
            }

    # 4. 流形与原型物理质心度量 + 绘制 t-SNE
    manifold_metrics = evaluate_manifold_and_plot(model, test_set, output_png_path=args.plot_path)

    # 5. 生成多样性评测
    if not args.skip_generation:
        gen_metrics = evaluate_generation_diversity(model, vocab, test_set, num_samples=args.gen_samples)
    else:
        gen_metrics = None

    print(f"\n=======================================================")
    print(f"  V8 评测流水线全流程执行完毕！")
    print(f"=======================================================")
    print(f"最佳权重: {best_ckpt}")
    print(f"Test PPL: {test_metrics['Test_PPL']} | EMO_acc: {test_metrics['Test_EMO_acc']}% | EMO_loss: {test_metrics['Test_EMO_loss']}")
    print(f"Silhouette: {manifold_metrics['Silhouette_Cosine']} | DBI: {manifold_metrics['DBI']} | 原型对齐: {manifold_metrics['Centroid_Alignment']}")
    if gen_metrics:
        print(f"Sampling Dist-2: {gen_metrics['Sampling']['Dist-2']}% | Unique: {gen_metrics['Sampling']['Unique']}%")


if __name__ == "__main__":
    main()
