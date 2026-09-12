# -*- coding: utf-8 -*-
"""
[CASE-EPCL 科研级特征流形与 PCAM 注意力可视化脚本]
功能:
1. 提取 V5 Trial 4b、V6 Trial 3、V6 Trial 4 在测试集上的情绪表征向量及原型向量
2. 计算定量聚类度量 (Silhouette Score, Davies-Bouldin, Calinski-Harabasz)
3. 绘制 t-SNE / PCA 流形散点对比图 (样本点分布 + 原型锚点)
4. 绘制 PCAM 跨步自回归原型注意力分配热力图 (Attention Heatmap)
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import shutil
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq
from src.utils.constants import ED_MAP_EMO
from src.models.common import get_input_from_batch

# 情绪反查字典
ID2EMO = {v: k for k, v in ED_MAP_EMO.items()}


def extract_features_and_prototypes(checkpoint_path, is_v6=True, num_samples=1000):
    """提取指定检查点的特征流形与原型矩阵"""
    print(f"[*] 正在从 {checkpoint_path} 提取特征与原型 (num_samples={num_samples})...")
    
    # 配置环境
    config.dataset = "ED"
    config.woStrategy = True
    config.num_prototypes_per_class = 2 if is_v6 else 1
    config.emotion_head_type = "residual_mlp" if is_v6 else "linear"
    config.use_pcam = is_v6
    config.test = True
    config.model = "case"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed()
    _, _, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=1)

    model = CASE(
        vocab,
        emotion_num=emo_num,
        strategy_num=strategy_num,
        is_eval=True,
        model_file_path=None,
    )
    model.to(config.device)

    state = torch.load(checkpoint_path, map_location=config.device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    all_features = []
    all_labels = []
    
    limit = min(num_samples, len(test_set.dataset) if hasattr(test_set, 'dataset') else num_samples)
    with torch.no_grad():
        for i, batch in enumerate(test_set):
            if i >= limit:
                break
            # 计算 react_enc
            (
                enc_batch,
                enc_lengths,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = get_input_from_batch(batch)
            src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            mask_emb = model.embedding(batch["mask_input"])
            src_emb = model.embedding(enc_batch) + mask_emb
            enc_outputs = model.encoder(src_emb, src_mask)

            # 情绪细粒度特征 fine_emotion 计算
            bsz, react_uttr_num, _ = batch["react_batch"].size()
            react_batch = batch["react_batch"].view(bsz * react_uttr_num, -1)
            react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
            react_emb = model.embedding(react_batch)
            react_batch_outputs = model.react_encoder(react_emb, react_batch_mask)
            react_batch_enc = torch.mean(react_batch_outputs, dim=1).view(bsz, react_uttr_num, -1)
            
            react_ctx_cat = torch.cat((
                react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)
            ), dim=-1).view(bsz * react_uttr_num, enc_outputs.size(1), -1)
            
            react_ctx_mask = src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz * react_uttr_num, -1, enc_outputs.size(1))
            react_batch_enc = model.react_ctx_encoder(react_ctx_cat, react_ctx_mask)[:, 0, :].view(bsz, react_uttr_num, -1)
            react_batch_enc = model.react_linear(react_batch_enc)
            fine_emotion = react_batch_enc[:, 0]  # [1, 300]
            label = batch["program_label"].item()

            all_features.append(fine_emotion.cpu().numpy()[0])
            all_labels.append(label)

        # 提取原型
        prototypes = model.epcl_criterion.prototypes.detach().cpu().numpy()

    return np.array(all_features), np.array(all_labels), prototypes, model, vocab, test_set


def plot_tsne_manifold(feats_dict, protos_dict, labels, save_dir, artifact_dir):
    """绘制对比 t-SNE 特征流形三联图"""
    print("[*] 正在执行 t-SNE 降维与聚类质量指标计算...")
    
    # 选取样本量较多、最具对比度的 8 种核心情绪作为清晰可视化子集
    selected_emotions = [0, 4, 8, 12, 16, 20, 24, 28]  # 覆盖 8 个代表性情感类别
    selected_mask = np.isin(labels, selected_emotions)
    sub_labels = labels[selected_mask]
    
    palette = plt.cm.get_cmap('tab10', len(selected_emotions))
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300)
    models_info = [
        ('V5 Trial 4b (ACF+BCF Base)', 'v5'),
        ('V6 Trial 3 (PCAM Baseline)', 'v6_t3'),
        ('V6 Trial 4 (Route B: HighDiv)', 'v6_t4')
    ]

    metrics_summary = {}

    for ax, (title, key) in zip(axes, models_info):
        X = feats_dict[key]
        P = protos_dict[key]
        
        # 计算全局 32 类完整聚类指标
        sil = silhouette_score(X, labels, metric='cosine')
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        metrics_summary[title] = {"Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch}
        
        # 仅对子集执行高质量 t-SNE 投影
        sub_X = X[selected_mask]
        
        # 如果是 V6 则每个类有 2 个原型，如果是 V5 则 1 个
        is_k2 = (P.shape[0] == 64)
        sub_proto_indices = []
        for c in selected_emotions:
            if is_k2:
                sub_proto_indices.extend([c * 2, c * 2 + 1])
            else:
                sub_proto_indices.append(c)
        sub_P = P[sub_proto_indices]

        # 联合降维以保持空间基底统一
        combined = np.vstack([sub_X, sub_P])
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(combined)
        
        X_2d = emb_2d[:len(sub_X)]
        P_2d = emb_2d[len(sub_X):]
        
        # 绘制样本点
        for idx, emo_id in enumerate(selected_emotions):
            mask = (sub_labels == emo_id)
            emo_name = ID2EMO.get(emo_id, f"Emo_{emo_id}")
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                color=palette(idx),
                alpha=0.45,
                s=24,
                label=f"{emo_name}" if key == 'v5' else None
            )
            
            # 绘制原型锚点 (大星号或五角星)
            if is_k2:
                p_pts = P_2d[idx*2 : idx*2+2]
            else:
                p_pts = P_2d[idx : idx+1]
            ax.scatter(
                p_pts[:, 0], p_pts[:, 1],
                color=palette(idx),
                edgecolors='black',
                linewidths=1.5,
                marker='*',
                s=180,
                zorder=10
            )

        ax.set_title(f"{title}\nSilhouette: {sil:.4f} | DB: {db:.2f}", fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.3)

    # 统一图例
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=8, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_png = os.path.join(save_dir, "tsne_manifold_comparison.png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"[OK] t-SNE 对比图已保存至: {out_png}")

    # 拷贝一份至 Artifact 目录
    art_png = os.path.join(artifact_dir, "tsne_manifold_comparison.png")
    shutil.copy2(out_png, art_png)
    print(f"[OK] 已同步至 Artifact 目录: {art_png}")

    return metrics_summary


def plot_pcam_attention_heatmap(model, test_set, save_dir, artifact_dir):
    """针对典型样本提取并绘制 PCAM 注意力热力图"""
    print("[*] 正在提取 PCAM 跨步注意力热力图...")
    model.eval()

    # 寻找一个具有鲜明情感特征的样本 (例如 label=joyful 或 terrified)
    target_batch = None
    for idx, b in enumerate(test_set):
        lbl = b["program_label"].item()
        if lbl in [8, 16, 20]:  # 选择典型情感
            target_batch = b
            break

    batch = target_batch
    emotion_name = ID2EMO.get(batch["program_label"].item(), "emotion")
    
    # 挂载 hook 捕获 PCAM 注意力权重
    attn_records = []
    def pcam_hook(module, input, output):
        dec_output, prototypes = input
        bsz, seq_len, _ = dec_output.size()
        k_protos, _ = prototypes.size()
        q = module.q_proj(dec_output) * module.scaling
        k = module.k_proj(prototypes)
        q = q.view(bsz, seq_len, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(k_protos, module.num_heads, module.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_weights, dim=-1)  # [1, num_heads, seq_len, 64]
        attn_records.append(attn_probs.detach().cpu())

    hook_handle = model.pcam.register_forward_hook(pcam_hook)

    # 运行一次单步自回归解码
    with torch.no_grad():
        sent = model.decoder_sampling(batch, temp=0.7, top_p=0.9, max_dec_step=15)
    
    hook_handle.remove()

    if not attn_records:
        print("[!] 未捕获到 PCAM 注意力")
        return

    # attn_records 中每个元素是每步的注意力，形状 [1, heads, 1, 64]
    all_step_attns = torch.cat(attn_records, dim=2)  # [1, heads, total_steps, 64]
    avg_head_attn = all_step_attns[0].mean(dim=0).numpy()  # [total_steps, 64]
    
    total_steps = avg_head_attn.shape[0]
    words = sent[0].split()[:total_steps]
    if len(words) < total_steps:
        words += ["<PAD>"] * (total_steps - len(words))

    plt.figure(figsize=(14, 5.5), dpi=300)
    plt.imshow(avg_head_attn, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title(f"PCAM Cross-Attention Distribution over 64 Sub-Prototypes\nEmotion: [{emotion_name}] | Generated: \"{' '.join(words)}\"", fontsize=12, fontweight='bold')
    plt.xlabel("64 Sub-Prototypes (32 Emotions x 2 Sub-Prototypes)", fontsize=11)
    plt.ylabel("Decoding Steps (Words)", fontsize=11)
    plt.yticks(range(len(words)), words, fontsize=10)
    
    # 高亮目标情绪的 2 个对应原型通道
    target_lbl = batch["program_label"].item()
    plt.axvline(x=target_lbl * 2, color='red', linestyle='--', alpha=0.7, label=f"Target Proto {target_lbl*2}")
    plt.axvline(x=target_lbl * 2 + 1, color='orange', linestyle='--', alpha=0.7, label=f"Target Proto {target_lbl*2+1}")
    plt.legend(loc='upper right')
    plt.tight_layout()

    out_png = os.path.join(save_dir, "pcam_attention_heatmap.png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"[OK] PCAM 注意力热力图已保存至: {out_png}")

    art_png = os.path.join(artifact_dir, "pcam_attention_heatmap.png")
    shutil.copy2(out_png, art_png)
    print(f"[OK] 已同步至 Artifact 目录: {art_png}")


if __name__ == '__main__':
    save_fig_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(save_fig_dir, exist_ok=True)
    artifact_dir = r"C:\Users\chy\.gemini\antigravity-ide\brain\3aaf26ff-c5f3-4539-8518-a7b639cdaf56"

    # 1. 提取 V5 Trial 4b
    v5_ckpt = "save/epcl_v5_trial4b/CASE_47999_37.8648"
    v5_feats, v5_labels, v5_protos, _, _, _ = extract_features_and_prototypes(v5_ckpt, is_v6=False, num_samples=800)

    # 2. 提取 V6 Trial 3
    v6_t3_ckpt = "save/epcl_v6_trial3/CASE_39999_36.8639"
    t3_feats, t3_labels, t3_protos, _, _, _ = extract_features_and_prototypes(v6_t3_ckpt, is_v6=True, num_samples=800)

    # 3. 提取 V6 Trial 4
    v6_t4_ckpt = "save/epcl_v6_trial4/CASE_33999_37.0487"
    t4_feats, t4_labels, t4_protos, model_t4, _, test_set = extract_features_and_prototypes(v6_t4_ckpt, is_v6=True, num_samples=800)

    feats_dict = {
        'v5': v5_feats,
        'v6_t3': t3_feats,
        'v6_t4': t4_feats,
    }
    protos_dict = {
        'v5': v5_protos,
        'v6_t3': t3_protos,
        'v6_t4': t4_protos,
    }

    # 4. 绘制 t-SNE 对比
    metrics = plot_tsne_manifold(feats_dict, protos_dict, v5_labels, save_fig_dir, artifact_dir)

    print("\n" + "="*80)
    print(f"{'模型架构版本':<30} | {'轮廓系数(Silhouette)↑':<20} | {'Davies-Bouldin↓':<15}")
    print("="*80)
    for k, v in metrics.items():
        print(f"{k:<30} | {v['Silhouette']:>18.4f} | {v['Davies-Bouldin']:>15.4f}")
    print("="*80)

    # 5. 绘制 PCAM 注意力热力图
    plot_pcam_attention_heatmap(model_t4, test_set, save_fig_dir, artifact_dir)
    print("\n[ALL DONE] 可视化全部完成！")
