### TAKEN FROM https://github.com/kolloldas/torchnlp
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
    MultiHeadAttention,
    Embeddings,
    Attention
)
from src.utils.config import config
from src.utils.constants import ESC_MAP_EMO, ESC_MAP_STRATEGY, ED_MAP_EMO

from sklearn.metrics import accuracy_score


# ================= V7 Architecture: 高维解缠残差投影头 (ERP) =================
class ExpandedResidualProjector(nn.Module):
    """
    [V7 Architecture] 高维解缠残差投影头 (Expanded Residual Projector, ERP)
    文献依据: SimCLR v2 (深层解耦), Barlow Twins (高维解缠扩张 300->768), SimCSE (残差直连保护语义先验)
    结构: Linear(300 -> 768) -> LN -> GELU -> Dropout -> Linear(768 -> 768) -> LN -> GELU -> Linear(768 -> 300)
          + Shortcut(Linear(300 -> 300) + LN) -> LN -> L2_Norm
    """
    def __init__(self, input_dim=300, hidden_dim=768, output_dim=300, dropout=0.1):
        super(ExpandedResidualProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # 残差直连捷径 (保护底层词法先验各向同性)
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.final_ln = nn.LayerNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        res = self.shortcut(x)
        h = self.dropout(self.act1(self.ln1(self.fc1(x))))
        h = self.act2(self.ln2(self.fc2(h)))
        out = self.fc3(h)
        return F.normalize(self.final_ln(out + res), p=2, dim=-1)


# ================= V7 Architecture: 动态上下文条件超网络原型 (CCHP) =================
class ContextConditionedHyperPrototypes(nn.Module):
    """
    [V7 Architecture] 动态上下文条件超网络原型 (Context-Conditioned Hyper-Prototypes, CCHP)
    理论依据: HyperNetworks (ICLR'17), Conditional ProtoNet (NeurIPS'17), FiLM (AAAI'18)
    机制: 保持全局基底原型锚定 32 类情绪宏观超球面几何拓扑，
          由对话历史上下文向量生成样本专属的语义位移场与通道缩放因子。
    """
    def __init__(self, num_classes=32, num_prototypes_per_class=2, input_dim=300,
                 hyper_dim=256, alpha_dyn=0.1):
        super(ContextConditionedHyperPrototypes, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.input_dim = input_dim
        self.alpha_dyn = alpha_dyn

        # 1. 全局静态基底原型
        self.prototypes = nn.Parameter(torch.empty(self.total_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

        # 2. 超网络位移场生成器 (双线性特征交互)
        self.hyper_net = nn.Sequential(
            nn.Linear(input_dim * 2, hyper_dim),
            nn.LayerNorm(hyper_dim),
            nn.GELU(),
            nn.Linear(hyper_dim, input_dim)
        )
        # 3. 门控尺度因子生成器
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.hyper_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.scale_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ctx_repr=None):
        """
        前向传播:
        ctx_repr: [B, D] 对话上下文全局表征向量。若为 None，则退化返回全局基底原型。
        返回:
        dyn_prototypes: 若 ctx_repr 为 None 则为 [M, D]，否则为 [B, M, D] (M = K*C)
        reg_loss: 位移场 L2 正则化标量损失 (防止超网络自指作弊与流形漂移)
        """
        p_base = F.normalize(self.prototypes, p=2, dim=-1)
        if ctx_repr is None:
            return p_base, torch.tensor(0.0, device=p_base.device)

        bsz = ctx_repr.size(0)
        M, D = p_base.size()

        p_base_exp = p_base.unsqueeze(0).expand(bsz, M, D)
        ctx_exp = ctx_repr.unsqueeze(1).expand(bsz, M, D)
        pair_input = torch.cat([p_base_exp, ctx_exp], dim=-1)  # [B, M, 2*D]

        delta_p = torch.tanh(self.hyper_net(pair_input))  # [B, M, D], 严格受限于 [-1, 1]
        scale = 2.0 * self.scale_net(pair_input)          # [B, M, D], 严格受限于 [0, 2]

        dyn_prototypes = F.normalize(scale * p_base_exp + self.alpha_dyn * delta_p, p=2, dim=-1)
        reg_loss = torch.mean(delta_p ** 2)
        return dyn_prototypes, reg_loss

def compute_unlikelihood_loss(log_probs, target_tokens, pad_idx=1):
    """
    [V7 Phase 2] 序列级无似然训练损失 (Sequence-level Repetition Unlikelihood Loss)
    针对自回归生成中高频出现的局部重复与安全套话，在训练端主动惩罚前文历史词元的预测概率。
    :param log_probs: [B, T, V] 词表对数概率分布 (log_softmax 或 generator 输出)
    :param target_tokens: [B, T] 真实标签序列
    :param pad_idx: 填充符号索引 (默认1)
    :return: 标量损失
    """
    B, T = target_tokens.size()
    if T <= 1:
        return torch.tensor(0.0, device=log_probs.device)

    # 1. 构造前文候选词元索引 [B, T_curr, T_hist]
    cand_tokens = target_tokens.unsqueeze(1).expand(B, T, T)
    # 限制索引在词表范围内
    cand_tokens_clamped = cand_tokens.clamp(0, log_probs.size(-1) - 1)
    cand_log_probs = torch.gather(log_probs, dim=-1, index=cand_tokens_clamped)

    # 2. 构造因果与非目标词元负样本掩码
    causal_mask = torch.tril(torch.ones(T, T, device=log_probs.device), diagonal=-1).bool().unsqueeze(0)
    not_curr_mask = (target_tokens.unsqueeze(2) != target_tokens.unsqueeze(1))
    not_pad_curr = target_tokens.ne(pad_idx).unsqueeze(2)
    not_pad_cand = target_tokens.ne(pad_idx).unsqueeze(1)

    final_mask = causal_mask & not_curr_mask & not_pad_curr & not_pad_cand

    if not final_mask.any():
        return torch.tensor(0.0, device=log_probs.device)

    selected_log_probs = cand_log_probs[final_mask]
    selected_probs = torch.clamp(torch.exp(selected_log_probs), max=0.9999)
    ul_loss = -torch.log(torch.clamp(1.0 - selected_probs, min=1e-7)).mean()
    return ul_loss


# ================= EPCL: 情感原型对比学习损失模块 =================
class PrototypeContrastiveLoss(nn.Module):
    """
    [V8 Upgraded] Multi-Prototype / MCP EPCL
    集成:
    1. 高维解缠残差投影头 (ERP)
    2. 动态上下文条件超网络原型 (CCHP)
    3. 动量样本中心原型 (MCP, Momentum Centroid Prototypes) [V8 核心]
    """
    def __init__(self, num_classes=32, input_dim=300, num_prototypes_per_class=1,
                 temperature=0.3, t_uniform=2.0, alpha_uni=1.0,
                 use_erp=False, erp_hidden_dim=768, erp_dropout=0.1,
                 use_cchp=False, cchp_alpha=0.1,
                 use_mcp=False, mcp_momentum=0.99,
                 use_arc_margin=False, arc_margin=0.30, arc_mode='cos'):
        super(PrototypeContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.total_prototypes = num_classes * num_prototypes_per_class
        self.temperature = temperature
        self.t_uniform = t_uniform
        self.alpha_uni = alpha_uni
        self.use_erp = use_erp
        self.use_cchp = use_cchp and not use_mcp
        self.use_mcp = use_mcp
        self.mcp_momentum = mcp_momentum
        self.use_arc_margin = use_arc_margin
        self.arc_margin = arc_margin
        self.arc_mode = arc_mode

        # 1. 投影头选择 (ERP vs 原版瓶颈 MLP)
        if self.use_erp:
            self.projection_head = ExpandedResidualProjector(
                input_dim=input_dim,
                hidden_dim=erp_hidden_dim,
                output_dim=input_dim,
                dropout=erp_dropout
            )
        else:
            proj_hidden = 128
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, input_dim)
            )

        # 2. 原型生成器选择 (MCP vs CCHP vs 静态 Parameter)
        if self.use_mcp:
            # [V8 手术 1] 动量样本中心原型 (MCP)：注册为不可导 Buffer，完全由样本几何重心驱动
            self.register_buffer("prototypes", torch.empty(self.total_prototypes, input_dim))
            nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)
            self.register_buffer("class_initialized", torch.zeros(self.total_prototypes, dtype=torch.bool))
        elif self.use_cchp:
            self.cchp = ContextConditionedHyperPrototypes(
                num_classes=num_classes,
                num_prototypes_per_class=num_prototypes_per_class,
                input_dim=input_dim,
                alpha_dyn=cchp_alpha
            )
        else:
            self.prototypes = nn.Parameter(torch.empty(self.total_prototypes, input_dim))
            nn.init.xavier_uniform_(self.prototypes)
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def __getattr__(self, name):
        if name == "prototypes":
            modules = self.__dict__.get('_modules', {})
            cchp = modules.get('cchp', None)
            if cchp is not None and hasattr(cchp, 'prototypes'):
                return cchp.prototypes
        return super(PrototypeContrastiveLoss, self).__getattr__(name)

    @property
    def current_prototypes(self):
        if self.use_mcp:
            return self.prototypes
        if self.use_cchp:
            return self.cchp.prototypes
        return self.prototypes

    def _update_mcp_prototypes(self, z, labels):
        """
        [V8 MCP] 动量样本中心原型 EMA 滚动更新
        z: [B, D] 经过投影头并已 L2 归一化的特征向量
        labels: [B] 真实情感类别标签
        """
        unique_labels = torch.unique(labels)
        new_prototypes = self.prototypes.clone()
        new_initialized = self.class_initialized.clone()

        for c in unique_labels:
            c_idx = c.item()
            if c_idx >= self.total_prototypes:
                continue
            mask_c = (labels == c)
            z_c = z[mask_c]  # [N_c, D]
            v_c = z_c.mean(dim=0)  # [D] 当前批次样本几何重心
            v_c_norm = F.normalize(v_c, p=2, dim=0)

            if not new_initialized[c_idx]:
                # 冷启动：首次捕获样本，直接赋给真实样本质心
                new_prototypes[c_idx] = v_c_norm
                new_initialized[c_idx] = True
            else:
                # EMA 指数平滑滚动更新: P_c = Normalize(μ * P_c + (1 - μ) * v_c)
                updated_p = self.mcp_momentum * new_prototypes[c_idx] + (1.0 - self.mcp_momentum) * v_c_norm
                new_prototypes[c_idx] = F.normalize(updated_p, p=2, dim=0)

        self.prototypes.copy_(new_prototypes)
        self.class_initialized.copy_(new_initialized)

    def uniformity_loss(self, normalized_prototypes):
        """原型超球面斥力正则"""
        if normalized_prototypes.dim() == 3:
            # [B, M, D] 批次动态原型: 在样本维度求平均
            B, M, D = normalized_prototypes.size()
            sq_pdist = 2.0 - 2.0 * torch.bmm(normalized_prototypes, normalized_prototypes.transpose(1, 2))
            mask = torch.eye(M, device=normalized_prototypes.device).unsqueeze(0).expand(B, M, M).bool()
            sq_pdist = sq_pdist.masked_fill(mask, float('inf'))
            return torch.logsumexp(-self.t_uniform * sq_pdist, dim=-1).mean()
        else:
            # [M, D] 静态原型
            sq_pdist = 2.0 - 2.0 * torch.matmul(normalized_prototypes, normalized_prototypes.T)
            mask = torch.eye(normalized_prototypes.size(0), device=normalized_prototypes.device).bool()
            sq_pdist = sq_pdist.masked_fill(mask, float('inf'))
            return torch.logsumexp(-self.t_uniform * sq_pdist, dim=1).mean()

    def forward(self, features, labels, tau=None, ctx_repr=None):
        current_tau = tau if tau is not None else self.temperature

        # 1. 投影特征
        if self.use_erp:
            proj_norm = self.projection_head(features)  # ERP 输出已 L2 归一化
        else:
            projected = self.projection_head(features)
            proj_norm = F.normalize(projected, p=2, dim=1)

        # 2. 原型获取与 [V8 MCP] 样本质心更新
        reg_loss = torch.tensor(0.0, device=features.device)
        if self.use_mcp:
            if self.training and labels is not None:
                with torch.no_grad():
                    self._update_mcp_prototypes(proj_norm, labels)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        elif self.use_cchp:
            proto_norm, reg_loss = self.cchp(ctx_repr=ctx_repr)
        else:
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)

        B = proj_norm.size(0)
        batch_idx = torch.arange(B, device=labels.device)

        # 3. 计算余弦相似度
        if proto_norm.dim() == 3:
            # [B, 1, D] x [B, D, M] -> [B, M]
            all_sim = torch.bmm(proj_norm.unsqueeze(1), proto_norm.transpose(1, 2)).squeeze(1)
        else:
            # [B, D] x [D, M] -> [B, M]
            all_sim = torch.matmul(proj_norm, proto_norm.T)

        if self.num_prototypes_per_class == 1:
            if getattr(self, "use_arc_margin", False) and self.arc_margin > 0.0 and self.training:
                # [V8 手术 3] 加性硬边际 Arc-EPCL：仅在训练期对目标正类别施加加性硬间隔惩罚
                margin_sim = all_sim.clone()
                target_sim = margin_sim[batch_idx, labels]
                if getattr(self, "arc_mode", "cos") == "angle":
                    # 角域加性硬边际: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
                    cos_theta = target_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp(min=1e-7))
                    m = self.arc_margin
                    cos_m = math.cos(m)
                    sin_m = math.sin(m)
                    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
                    # 阈值防反转保护 (cos_theta > cos(pi - m) = -cos(m))
                    threshold = math.cos(math.pi - m)
                    margin_sim[batch_idx, labels] = torch.where(
                        cos_theta > threshold,
                        cos_theta_m,
                        cos_theta - math.sin(m) * m
                    )
                else:
                    # 余弦加性硬边际 (CosFace 形式): cos(θ) - m，平滑且极度数值稳定
                    margin_sim[batch_idx, labels] = target_sim - self.arc_margin
                logits = margin_sim / current_tau
            else:
                logits = all_sim / current_tau
            loss_align = F.cross_entropy(logits, labels)
        else:
            K = self.num_prototypes_per_class
            C = self.num_classes
            all_cosine = all_sim.view(B, C, K)

            target_cosines = all_cosine[batch_idx, labels]
            best_pos_cosine, _ = target_cosines.max(dim=-1, keepdim=True)

            neg_mask = torch.ones(B, C, dtype=torch.bool, device=labels.device)
            neg_mask[batch_idx, labels] = False
            neg_cosines = all_cosine[neg_mask].view(B, (C - 1) * K)

            if getattr(self, "use_arc_margin", False) and self.arc_margin > 0.0 and self.training:
                if getattr(self, "arc_mode", "cos") == "angle":
                    cos_theta = best_pos_cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                    sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp(min=1e-7))
                    m = self.arc_margin
                    cos_m = math.cos(m)
                    sin_m = math.sin(m)
                    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
                    threshold = math.cos(math.pi - m)
                    best_pos_cosine = torch.where(
                        cos_theta > threshold,
                        cos_theta_m,
                        cos_theta - math.sin(m) * m
                    )
                else:
                    best_pos_cosine = best_pos_cosine - self.arc_margin

            comp_logits = torch.cat([best_pos_cosine, neg_cosines], dim=-1) / current_tau
            target_zeros = torch.zeros(B, dtype=torch.long, device=labels.device)
            loss_align = F.cross_entropy(comp_logits, target_zeros)

        loss_uni = self.uniformity_loss(proto_norm)
        return loss_align + self.alpha_uni * loss_uni, reg_loss


# ================= V6 Trial 1: 双层非线性残差情感分类头 =================
class ResidualEmotionHead(nn.Module):
    """
    双层非线性残差情感分类头 (Residual Emotion Head):
    结构: LayerNorm -> Linear(d_in, d_hid) -> GELU -> Dropout -> Linear(d_hid, num_classes)
          + Linear(d_in, num_classes) 残差直连捷径 (Shortcut)
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super(ResidualEmotionHead, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.res_proj = nn.Linear(input_dim, num_classes)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.res_proj.weight)
        nn.init.zeros_(self.res_proj.bias)

    def forward(self, x):
        norm_x = self.norm(x)
        h = self.fc2(self.dropout(self.act(self.fc1(norm_x))))
        res = self.res_proj(x)
        return h + res


# ================= PCAM: 原型交叉记忆注意力模块 (支持动态与静态原型) =================
class PrototypeConditionedAttentionModule(nn.Module):
    """
    原型交叉记忆注意力模块 (Prototype-Conditioned Attention Module, PCAM):
    定位: 解码器隐状态输出端与生成器之间的自适应门控残差跨注意力桥梁。
    [V7 增强]: 原生兼容 2D 静态原型 [M, D] 与 3D 动态上下文原型 [B, M, D]。
    """
    def __init__(self, d_model=300, num_heads=2, dropout=0.1, gate_bias=-1.0):
        super(PrototypeConditionedAttentionModule, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.gate_linear = nn.Linear(2 * d_model, d_model)
        nn.init.constant_(self.gate_linear.bias, gate_bias)
        nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.1)

        self.layer_norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, dec_output, prototypes):
        """
        前向传播:
        dec_output: [bsz, seq_len, d_model] 解码器输出隐状态
        prototypes: [k_protos, d_model] 或 [bsz, k_protos, d_model] 原型张量
        返回: enhanced_output [bsz, seq_len, d_model]
        """
        bsz, seq_len, _ = dec_output.size()

        # 1. 计算 Query
        q = self.q_proj(dec_output) * self.scaling
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, heads, seq_len, head_dim]

        # 2. 计算 Key, Value (自适应 2D / 3D 原型)
        if prototypes.dim() == 2:
            k_protos, _ = prototypes.size()
            k = self.k_proj(prototypes).view(k_protos, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
            v = self.v_proj(prototypes).view(k_protos, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0).expand(bsz, -1, -1, -1)
        else:
            _, k_protos, _ = prototypes.size()
            k = self.k_proj(prototypes).view(bsz, k_protos, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(prototypes).view(bsz, k_protos, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. 计算注意力权重 [bsz, heads, seq_len, k_protos]
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 4. 计算原型上下文表征
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        h_proto = self.out_dropout(self.out_proj(context))

        # 5. 通道自适应门控与残差融合
        gate_input = torch.cat([dec_output, h_proto], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        enhanced_output = self.layer_norm(dec_output + gate * h_proto)
        return enhanced_output
# ==============================================================================


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask, cs_enc_outputs, cs_enc_mask, concept_enc_outputs, concept_enc_mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask), cs_enc_outputs, cs_enc_mask, concept_enc_outputs, concept_enc_mask))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)

class RelationMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, weights_dropout=True):
        super(RelationMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == self.emb_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.query_linear = nn.Linear(emb_dim, emb_dim)
        self.key_linear = nn.Linear(emb_dim, emb_dim)
        self.value_linear = nn.Linear(emb_dim, emb_dim)
        self.relation_linear = nn.Linear(emb_dim, 2*emb_dim, bias=False)
        
        self.output_linear = nn.Linear(emb_dim, emb_dim)
        self.weights_dropout = weights_dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_linear.weight, std=0.02)
        nn.init.normal_(self.key_linear.weight, std=0.02)
        nn.init.normal_(self.value_linear.weight, std=0.02)
        nn.init.normal_(self.relation_linear.weight, std=0.02)
        nn.init.normal_(self.output_linear.weight, std=0.02)
        nn.init.constant_(self.query_linear.bias, 0.)
        nn.init.constant_(self.key_linear.bias, 0.)
        nn.init.constant_(self.value_linear.bias, 0.)
        nn.init.constant_(self.output_linear.bias, 0.)
    
    def forward(self, queries, keys, values, adj_mask, relation):
        bsz, tgt_len, emb_dim = queries.size()
        src_len = keys.size(1)
        
        queries = self.query_linear(queries) * self.scaling
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        queries = queries.view(bsz, tgt_len, self.num_heads, self.head_dim)
        keys = keys.view(bsz, src_len, self.num_heads, self.head_dim)
        values = values.view(bsz, src_len, self.num_heads, self.head_dim)
        
        relation_queries, relation_keys = self.relation_linear(relation).chunk(2, dim=-1)
        relation_queries = relation_queries.view(bsz, tgt_len, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        relation_keys = relation_keys.view(bsz, tgt_len, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        queries = queries.unsqueeze(2) + relation_queries
        keys = keys.unsqueeze(1) + relation_keys
        
        attn_weights = torch.einsum('bijhn,bijhn->bijh', [queries, keys])
        assert list(attn_weights.size()) == [bsz, tgt_len, src_len, self.num_heads]
        
        adj_mask = adj_mask.data.eq(config.PAD_idx)
        attn_weights.masked_fill_(
            adj_mask.unsqueeze(-1),
            float("-inf")
        )
        
        attn_weights = F.softmax(attn_weights, dim=2)
        
        if self.weights_dropout:
            attn_weights = self.attn_dropout(attn_weights)
        
        # attn_weights: bsz x tgt_len x src_len x heads
        # values: bsz x src_len x heads x dim
        outputs = torch.einsum('bijh,bjhn->bihn', [attn_weights, values]).contiguous()

        if not self.weights_dropout:
            outputs = self.attn_dropout(outputs)
        
        assert list(outputs.size()) == [bsz, tgt_len, self.num_heads, self.head_dim]
        
        outputs = self.output_linear(outputs.view(bsz, tgt_len, self.emb_dim))
        
        return outputs, attn_weights

class GraphTransformer(nn.Module):
    def __init__(self, layer_num, emb_dim, ffn_emb_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layer_num):
            self.layers.append(GraphTransformerLayer(emb_dim, ffn_emb_dim, num_heads, dropout, weights_dropout))
    
    def forward(self, queries, keys, values, adj_mask, relation=None):
        for _, layer in enumerate(self.layers):
            outputs, _ = layer(queries, keys, values, adj_mask, relation)
        return outputs

class GraphTransformerLayer(nn.Module):
    def __init__(self, emb_dim, ffn_emb_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(emb_dim, emb_dim, emb_dim, emb_dim, num_heads=num_heads, dropout=dropout)
        self.relation_multi_head_attention = RelationMultiHeadAttention(emb_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(emb_dim, ffn_emb_dim)
        self.fc2 = nn.Linear(ffn_emb_dim, emb_dim)
        
        self.attn_layer_norm = LayerNorm(emb_dim)
        self.ffn_layer_norm = LayerNorm(emb_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
    
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def forward(self, queries, keys, values, adj_mask, relation=None):
        residual = queries
        
        if relation is None:
            outputs, attn_weights = self.multi_head_attention(queries, keys, values, adj_mask)
        else:
            outputs, attn_weights = self.relation_multi_head_attention(queries, keys, values, adj_mask, relation)
        
        outputs = self.attn_dropout(outputs)
        outputs = self.attn_layer_norm(residual + outputs)
        
        residual = outputs 
        outputs = self.fc2(self.gelu(self.fc1(outputs)))
        outputs = self.ffn_dropout(outputs)
        outputs = self.ffn_layer_norm(residual + outputs)
        return outputs, attn_weights

def create_embedding(num, emb_dim, padding_idx=None):
    """Create and initialize embeddings."""
    embedding = Embeddings(num, emb_dim, padding_idx=padding_idx)
    nn.init.normal_(embedding.lut.weight, mean=0, std=emb_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(embedding.lut.weight[padding_idx], 0)
    return embedding

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, prior_enc, poster_enc):
        prior_enc = self.sigm(prior_enc)
        poster_enc = self.sigm(poster_enc)
        score = torch.squeeze(self.bilinear(prior_enc, poster_enc), 1)
        return self.sigm(score)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 3 if config.woStrategy else 4
        input_dim = input_num * config.hidden_dim
        hid_num = 1 if config.woStrategy else 2
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask):
        N = h.shape[0]
        assert self.dim == h.shape[2]
        mask=-1e30*mask.float()

        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        attention = F.softmax(e+mask.unsqueeze(-1),dim=1)
        return torch.matmul(torch.transpose(attention,1,2), h).squeeze(1),attention

class CASE(nn.Module):
    def __init__(
        self,
        vocab,
        emotion_num,
        strategy_num,
        model_file_path=None,
        is_eval=False,
        load_optim=False
    ):
        super(CASE, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.dataset = config.dataset
        
        self.word_freq = np.zeros(self.vocab_size)
        
        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
        
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.relation_embedding = create_embedding(
            config.relation_num, 
            emb_dim=config.emb_dim, 
            padding_idx=config.PAD_idx
        )
        
        # Context encoder
        self.encoder = self.make_encoder(config.emb_dim)
        # Cognition
        self.cognition_encoder = self.make_encoder(config.emb_dim)
        self.cs_graph_encoder = GraphTransformer(
            config.graph_layer_num, 
            config.emb_dim, 
            config.graph_ffn_emb_dim, 
            config.graph_num_heads, 
            config.dropout
        )
        self.cs_prior_attn = Attention(query_size=config.emb_dim,
                                       memory_size=config.emb_dim,
                                       hidden_size=config.emb_dim,
                                       mode="dot"
                                    )
        self.cs_posterior_attn = Attention(query_size=config.emb_dim,
                                       memory_size=config.emb_dim,
                                       hidden_size=config.emb_dim,
                                       mode="dot"
                                    )
        # Affection
        self.react_encoder = self.make_encoder(config.emb_dim)
        self.react_selfattn = SelfAttention(config.emb_dim, config.emb_dim)
        self.react_ctx_encoder = self.make_encoder(2 * config.emb_dim)
        self.react_linear = nn.Linear(config.emb_dim, config.emb_dim)
        self.fine_emotion_selfattn = SelfAttention(config.emb_dim, config.emb_dim)
        self.emotion_gate = nn.Linear(config.emb_dim, 1)
        self.emotion_norm = nn.Linear(2*config.emb_dim, config.emb_dim)
        self.concept_graph_encoder = GraphTransformer(
            config.graph_layer_num, 
            config.emb_dim, 
            config.graph_ffn_emb_dim, 
            config.graph_num_heads, 
            config.dropout
        )
        self.vad_layernorm = nn.LayerNorm(config.emb_dim)
        if self.dataset == "ESConv":
            self.emotion_num = len(ESC_MAP_EMO)
        else:
            self.emotion_num = len(ED_MAP_EMO)
        
        # [V6 Trial 1] 情感分类头支持原版 linear 与全新 residual_mlp 双层非线性残差头
        if getattr(config, "emotion_head_type", "linear") == "residual_mlp":
            self.emotion_linear = ResidualEmotionHead(
                input_dim=config.emb_dim,
                hidden_dim=getattr(config, "mlp_hidden_dim", 300),
                num_classes=self.emotion_num,
                dropout=getattr(config, "mlp_dropout", 0.1)
            )
            print(f"[V6 Architecture] 成功启用双层非线性残差分类头 (ResidualEmotionHead): in={config.emb_dim}, hid={getattr(config, 'mlp_hidden_dim', 300)}, out={self.emotion_num}, dropout={getattr(config, 'mlp_dropout', 0.1)}")
        else:
            self.emotion_linear = nn.Linear(config.emb_dim, self.emotion_num)
        
        # === EPCL Module Init ===
        # [V6 Trial 2 / V7] Multi-Prototype: K sub-prototypes per class
        self.num_prototypes_per_class = getattr(config, 'num_prototypes_per_class', 1)
        total_protos = self.emotion_num * self.num_prototypes_per_class
        self.use_erp = getattr(config, 'use_erp', False)
        self.use_cchp = getattr(config, 'use_cchp', False)
        self.use_mcp = getattr(config, 'use_mcp', False)
        self.mcp_momentum = getattr(config, 'mcp_momentum', 0.99)
        self.use_arc_margin = getattr(config, 'use_arc_margin', False)
        self.arc_margin = getattr(config, 'arc_margin', 0.30)
        self.arc_mode = getattr(config, 'arc_mode', 'cos')
        self.epcl_anchor = getattr(config, 'epcl_anchor', 'fine_emotion')
        self.epcl_criterion = PrototypeContrastiveLoss(
            self.emotion_num, config.emb_dim,
            num_prototypes_per_class=self.num_prototypes_per_class,
            alpha_uni=getattr(config, 'alpha_uni', 1.0),
            use_erp=self.use_erp,
            erp_hidden_dim=getattr(config, 'erp_hidden_dim', 768),
            erp_dropout=getattr(config, 'erp_dropout', 0.1),
            use_cchp=self.use_cchp and not self.use_mcp,
            cchp_alpha=getattr(config, 'cchp_alpha', 0.1),
            use_mcp=self.use_mcp,
            mcp_momentum=self.mcp_momentum,
            use_arc_margin=self.use_arc_margin,
            arc_margin=self.arc_margin,
            arc_mode=self.arc_mode
        )
        if self.use_mcp:
            print(f"[V8 Architecture] 成功启用动量样本中心原型 (MCP): momentum={self.mcp_momentum}")
        elif self.use_cchp:
            print(f"[V7 Architecture] 成功启用动态上下文条件超网络原型 (CCHP): alpha_dyn={getattr(config, 'cchp_alpha', 0.1)}")
        if self.use_arc_margin:
            print(f"[V8 Architecture] 成功启用加性角度硬边际对比损失 (Arc-EPCL): margin={self.arc_margin}, mode={self.arc_mode}")
        self.cls_anchor = getattr(config, 'cls_anchor', 'default')
        if self.cls_anchor == 'default':
            self.cls_anchor = self.epcl_anchor
        self.soft_freeze = getattr(config, 'soft_freeze', False)
        self.freeze_decay_weight = getattr(config, 'freeze_decay_weight', 0.05)
        if self.soft_freeze:
            print(f"[V8 Trial 4] 成功启用分类头时序软退火 (Soft Freeze): decay_weight={self.freeze_decay_weight}")
        print(f"[V8 Trial 4] 情感分类头特征锚点: {self.cls_anchor} | EPCL 对比学习特征锚点: {self.epcl_anchor}")
        if self.use_erp:
            print(f"[V7 Architecture] 成功启用高维解缠残差投影头 (ERP): hidden={getattr(config, 'erp_hidden_dim', 768)}, dropout={getattr(config, 'erp_dropout', 0.1)}")
        if self.num_prototypes_per_class > 1:
            print(f"[V6/V7 Multi-Prototype] EPCL: {self.emotion_num} classes x {self.num_prototypes_per_class} sub-protos = {total_protos} total prototypes")
        self.emo_dropout = nn.Dropout(0.3)
        # [V4 Trial 8: MoP-DR] Router input dim = total_protos (32 when K=1, 64 when K=2)
        self.router_linear = nn.Linear(total_protos, config.emb_dim)
        self.concept_prior_attn = Attention(query_size=config.emb_dim,
                                            memory_size=config.emb_dim,
                                            hidden_size=config.emb_dim,
                                            mode="dot"
                                        )
        self.concept_posterior_attn = Attention(query_size=config.emb_dim,
                                                memory_size=config.emb_dim,
                                                hidden_size=config.emb_dim,
                                                mode="dot"
                                            )
        # Alignment
        self.bow_output_layer = nn.Sequential(
                    nn.Linear(in_features=2*config.emb_dim, out_features=config.emb_dim),
                    nn.Tanh(),
                    nn.Linear(in_features=config.emb_dim, out_features=self.vocab_size),
                    nn.LogSoftmax(dim=-1))
        # self.discriminator = Discriminator(config.emb_dim)
        
        # Strategy
        if self.dataset == "ESConv":
            self.strategy_encoder = self.make_encoder(config.emb_dim)
            self.strategy_num = len(ESC_MAP_STRATEGY)
            self.strategy_embedding = create_embedding(
                self.strategy_num, 
                emb_dim=config.emb_dim,
                padding_idx=config.PAD_idx
            )
            self.position_embedding = create_embedding(
                self.strategy_num, 
                emb_dim=config.emb_dim,
                padding_idx=None
            )
            self.strategy_layernorm = LayerNorm(config.emb_dim)
            self.strategy_linear = nn.Linear(config.emb_dim, self.strategy_num)
            self.prior_query_linear = nn.Linear(2*config.emb_dim, config.emb_dim)
        else:
            self.prior_query_linear = nn.Linear(config.emb_dim, config.emb_dim)
        
        # Merge
        self.ctx_merge_lin = MLP()
        
        # Decoder
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )
        
        # [V6 Trial 3] 原型交叉记忆注意力 (PCAM) 模块初始化
        self.use_pcam = getattr(config, 'use_pcam', False)
        if self.use_pcam:
            self.pcam = PrototypeConditionedAttentionModule(
                d_model=config.hidden_dim,
                num_heads=getattr(config, 'pcam_heads', 2),
                dropout=getattr(config, 'pcam_dropout', 0.1),
                gate_bias=getattr(config, 'pcam_gate_bias', -1.0)
            )
            print(f"[V6 Architecture] 成功启用原型交叉记忆注意力模块 (PCAM): heads={getattr(config, 'pcam_heads', 2)}, dropout={getattr(config, 'pcam_dropout', 0.1)}, gate_bias={getattr(config, 'pcam_gate_bias', -1.0)}")

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)
        
        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight
        
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        self.criterion.weight = torch.ones(self.vocab_size)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.criterion_kl = nn.KLDivLoss(reduction="mean")
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_bce_fine = nn.BCEWithLogitsLoss(reduction="none")
        # self.criterion_bce = nn.BCELoss()
        # self.criterion_bcs_fine = nn.BCELoss(reduction="none")
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bow = nn.NLLLoss(ignore_index=config.PAD_idx, reduction='mean')
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                config.warmup,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )
        
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        # [V5 Trial 4: 自适应分类头冻结状态]
        self.is_frozen = False
        self.actual_freeze_step = config.epcl_freeze_step

    def freeze_emo_head(self, step, best_state=None, best_step=None):
        """[V5 Trial 4b / V8 Trial 4: ACF-BCF] 冻结/软退火情感分类头，支持回滚至验证集历史最佳泛化权重并将退火起点锚定至 step"""
        if not self.is_frozen and self.dataset == "ED":
            if best_state is not None:
                self.emotion_linear.load_state_dict(best_state)
                print(f"[Adaptive Freeze + BCF] 成功将分类头 emotion_linear 权重回滚至历史最佳状态 (Step {best_step})！")
            if not getattr(config, 'soft_freeze', False):
                for p in self.emotion_linear.parameters():
                    p.requires_grad = False
                print(f"[Adaptive Freeze] Step {step}: 分类头 emotion_linear 已物理硬冻结，退火起点更新为 Step {step}！")
            else:
                for p in self.emotion_linear.parameters():
                    p.requires_grad = True
                print(f"[Adaptive Soft Freeze] Step {step}: 启用分类头时序软退火 (Soft Freeze, decay={getattr(config, 'freeze_decay_weight', 0.05)})，分类头参数保持温和微调更新！")
            self.is_frozen = True
            self.actual_freeze_step = step

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def apply_pcam(self, dec_output, ctx_repr=None):
        """
        [V6/V7 Architecture: PCAM] 统一对解码器输出隐状态注入原型交叉记忆
        dec_output: [bsz, seq_len, d_model] 解码器输出隐状态
        ctx_repr: [bsz, d_model] 可选上下文表征 (用于 CCHP 动态原型生成)
        返回: 经 PCAM 增强后的隐状态 (若未启用 PCAM 则返回原张量)
        """
        if getattr(self, "use_pcam", False) and hasattr(self, "pcam") and hasattr(self, "epcl_criterion"):
            if getattr(self.epcl_criterion, "use_cchp", False) and ctx_repr is not None:
                protos, _ = self.epcl_criterion.cchp(ctx_repr=ctx_repr)
            else:
                protos = self.epcl_criterion.current_prototypes
            return self.pcam(dec_output, protos)
        return dec_output

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "CASE_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)
    
    def add_position_embedding(self, sequence, seq_embeddings):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = seq_embeddings + position_embeddings
        sequence_emb = self.strategy_layernorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb
    
    def infomax(self, cs_enc, cs_fake_enc, concept_enc, concept_fake_enc):
        cs_enc = self.dropout(cs_enc)
        concept_enc = self.dropout(concept_enc)
        logits = self.discriminator(cs_enc, concept_enc)
        
        cs_fake_enc = self.dropout(cs_fake_enc)
        cs_fake_logits = self.discriminator(cs_fake_enc, concept_enc)
        
        concept_fake_enc = self.dropout(concept_enc)
        concept_fake_logits = self.discriminator(cs_enc, concept_fake_enc)
        
        mim_logits = torch.cat((logits, cs_fake_logits, concept_fake_logits))
        mim_lables = torch.cat((torch.ones_like(logits),
                                torch.zeros_like(cs_fake_logits),
                                torch.zeros_like(concept_fake_logits)))
        mim_loss = self.criterion_bce(mim_logits, mim_lables)
        return mim_loss
    
    def infomax_score(self, cs_enc, cs_fake_enc, concept_enc, concept_fake_enc):
        pos_score = torch.mul(cs_enc, concept_enc)
        logits = torch.sigmoid(torch.sum(pos_score, -1))
        cs_fake_score = torch.mul(cs_fake_enc, concept_enc)
        cs_fake_logits = torch.sigmoid(torch.sum(cs_fake_score, -1))
        concept_fake_score = torch.mul(cs_enc, concept_fake_enc)
        concept_fake_logits = torch.sigmoid(torch.sum(concept_fake_score, -1))
        cs_distance = torch.sigmoid(logits - cs_fake_logits)
        cs_loss = self.criterion_bce(cs_distance, torch.ones_like(cs_distance, dtype=torch.float32))
        concept_distance = torch.sigmoid(logits - concept_fake_logits)
        concept_loss = self.criterion_bce(concept_distance, torch.ones_like(concept_distance, dtype=torch.float32))
        mim_loss = cs_loss + concept_loss
        return mim_loss
    
    def fine_grained_infomax_score(self, cs_enc, cs_fake_enc, react_enc, react_fake_enc, mask):
        bsz, num, _ = cs_enc.size()
        cs_enc = cs_enc.reshape(bsz*num, -1)
        cs_fake_enc = cs_fake_enc.reshape(bsz*num, -1)
        react_enc = react_enc.view(bsz*num, -1)
        react_fake_enc = react_fake_enc.view(bsz*num, -1)
       
        pos_score = torch.mul(cs_enc, react_enc)
        logits = torch.sigmoid(torch.sum(pos_score, -1))
        
        cs_fake_score = torch.mul(cs_enc, react_fake_enc)
        cs_fake_logits = torch.sigmoid(torch.sum(cs_fake_score, -1))
        cs_distance = torch.sigmoid(logits - cs_fake_logits)
        cs_loss = self.criterion_bce_fine(cs_distance, torch.ones_like(cs_distance, dtype=torch.float32))
        cs_loss = torch.sum(cs_loss * mask.flatten()) / torch.sum(mask.flatten())
        
        react_fake_score = torch.mul(cs_fake_enc, react_enc)
        react_fake_logits = torch.sigmoid(torch.sum(react_fake_score, -1))
        react_distance = torch.sigmoid(logits - react_fake_logits)
        react_loss = self.criterion_bce_fine(react_distance, torch.ones_like(react_distance, dtype=torch.float32))
        react_loss = torch.sum(react_loss * mask.flatten()) / torch.sum(mask.flatten())
        
        fine_mim_loss = cs_loss + react_loss
        return fine_mim_loss
    
    def train_one_batch(self, batch, iter, train=True):
        accum_steps = 4
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_vad_batch = batch["context_vad"]
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        
        # [V5 Trial 4] 确定当前是否已处于冻结期（支持自适应 ACF 机制与固定步数回退）
        if config.adaptive_freeze:
            is_currently_frozen = self.is_frozen
            # 若开启自适应，但步数已达 max_freeze_step 且仍未冻结，强制自适应冻结兜底
            if train and not self.is_frozen and iter >= config.max_freeze_step and self.dataset == "ED":
                self.freeze_emo_head(iter)
                is_currently_frozen = True
        else:
            is_currently_frozen = (iter >= config.epcl_freeze_step)
            if train and iter == config.epcl_freeze_step and self.dataset == "ED":
                for p in self.emotion_linear.parameters():
                    p.requires_grad = False
                self.is_frozen = True
                self.actual_freeze_step = config.epcl_freeze_step
                print(f"[EPCL] Step {iter}: 分类头 emotion_linear 已冻结")
        
        if iter % accum_steps == 0:
            if config.noam:
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            # Encode Context
            src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
            mask_emb = self.embedding(batch["mask_input"])
            src_emb = self.embedding(enc_batch) + mask_emb
            enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        
            # Affection: Encode Concept
            concept_input = batch["concept_batch"]
            concept_mask = concept_input.data.eq(config.PAD_idx).unsqueeze(1)
            concept_vad_batch = batch["concept_vad_batch"]
            mask_concept = batch["mask_concept"]
            concept_adj_mask = batch["concept_adjacency_mask_batch"]
            # mask_concept = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
            # concept_mask = self.embedding(mask_concept)  # KG_idx embedding
            concept_emb = self.embedding(concept_input) + self.embedding(mask_concept)  # KG_idx embedding
            src_concept_input_emb = torch.cat((src_emb, concept_emb), dim=1)
            src_concept_outputs = self.concept_graph_encoder(src_concept_input_emb, 
                                                             src_concept_input_emb,
                                                             src_concept_input_emb,
                                                             concept_adj_mask)
            src_concept_mask = torch.cat((enc_batch, concept_input), dim=1).data.eq(config.PAD_idx)
            src_concept_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)
            src_concept_vad = torch.softmax(src_concept_vad, dim=-1).unsqueeze(2).repeat(1, 1, config.emb_dim)
            src_concept_outputs = self.vad_layernorm(src_concept_vad * src_concept_outputs)
        
            # Cognition: Encode Commonsense
            bsz, uttr_num, uttr_length = batch["uttr_batch_concat"].size()
            uttr_batch_concat = batch["uttr_batch_concat"].view(bsz*uttr_num, -1)
            uttr_batch_mask = uttr_batch_concat.data.eq(config.PAD_idx).unsqueeze(1)
        
            bsz, cs_num, cs_length = batch["cs_batch"].size()
            cs_batch = batch["cs_batch"].view(bsz*cs_num, -1)
            cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
            cs_mask = batch["cs_mask"] # bsz, cs_num
        
            cs_adj_mask = batch["cs_adjacency_mask_batch"]
        
            uttr_batch_emb = self.embedding(uttr_batch_concat)
            uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_batch_mask)[:,0].view(bsz, uttr_num, -1)
        
            cs_batch_emb = self.embedding(cs_batch)
            cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)[:,0].view(bsz, cs_num, -1)
            uttr_cs_outputs = torch.cat((uttr_batch_outputs, cs_batch_outputs), dim=1)
        
            assert uttr_cs_outputs.size(1) == cs_adj_mask.size(1)
            relation_emb = self.relation_embedding(cs_adj_mask)
            uttr_cs_graph_outputs = self.cs_graph_encoder(uttr_cs_outputs,
                                                      uttr_cs_outputs,
                                                      uttr_cs_outputs,
                                                      cs_adj_mask,
                                                      relation_emb)
        
            commonsense_outputs = uttr_cs_graph_outputs[:,-cs_batch_outputs.size(1):,:]
            commonsense_mask = cs_mask
        
            # Strategy: Encode Strategy Sequence
            if self.dataset == "ESConv":
                strategy_seqs = batch["strategy_seqs_batch"]
                mask_strategy = strategy_seqs.data.eq(config.PAD_idx).unsqueeze(1)
                strategy_seqs_emb = self.strategy_embedding(strategy_seqs)
                strategy_seqs_emb = self.add_position_embedding(strategy_seqs, strategy_seqs_emb)
                strategy_enc_outputs = self.strategy_encoder(strategy_seqs_emb, mask_strategy)
                strategy_enc_outputs = strategy_enc_outputs[:,0,:]
            
                prior_query = self.tanh(self.prior_query_linear(torch.cat((enc_outputs[:,0,:], strategy_enc_outputs), dim=-1)))
            else:
                prior_query = self.tanh(self.prior_query_linear(enc_outputs[:,0,:]))
        
            # concept
            prior_concept_enc, prior_concept_attn = self.concept_prior_attn(
                query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
                memory = self.tanh(src_concept_outputs),
                mask = src_concept_mask
            )
            prior_concept_attn = prior_concept_attn.squeeze(1)
        
            # commonsense
            prior_cs_enc, prior_cs_attn = self.cs_prior_attn(
                query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
                memory = self.tanh(commonsense_outputs),
                mask = commonsense_mask.eq(0)
            )
            prior_cs_attn = prior_cs_attn.squeeze(1)
        
            # knowledge selection
            target_post_batch = batch["target_post_batch"]
            mask_target = target_post_batch.data.eq(config.PAD_idx).unsqueeze(1)
            target_emb = self.embedding(target_post_batch) + self.embedding(batch["target_post_mask"])
            target_cs_outputs = self.cognition_encoder(target_emb, mask_target)[:, 0, :]
            target_concept_outputs = self.encoder(target_emb, mask_target)[:, 0, :]
            # merge strategy
        
            # concept selection
            posterior_concept_enc, posterior_concept_attn = self.concept_posterior_attn(
                query = target_concept_outputs.unsqueeze(1),
                memory = self.tanh(src_concept_outputs),
                mask = src_concept_mask
            )
            posterior_concept_attn = posterior_concept_attn.squeeze(1)
        
            # commonsense selection
            posterior_cs_enc, posterior_cs_attn = self.cs_posterior_attn(
                query = target_cs_outputs.unsqueeze(1),
                memory = self.tanh(commonsense_outputs),
                mask = commonsense_mask.eq(0)
            )
            posterior_cs_attn = posterior_cs_attn.squeeze(1)
        
            # pretrain bow loss
            bow_logits = self.bow_output_layer(torch.cat((posterior_concept_enc, posterior_cs_enc), dim=-1)) # bsz, 1, vocab_size
            bow_logits = bow_logits.repeat(1, dec_batch.size(1), 1)
            bow_loss = self.criterion_bow(
                bow_logits.contiguous().view(-1, bow_logits.size(-1)), 
                dec_batch.contiguous().view(-1)
            )
        
            if config.pretrain and train:
                bow_loss.backward()
                self.optimizer.step()
                return bow_loss.item()
        
            # distribution alignment
            concept_kl_loss = self.criterion_kl(torch.log(prior_concept_attn+1e-20),
                                                posterior_concept_attn.detach())
            cs_kl_loss = self.criterion_kl(torch.log(prior_cs_attn+1e-20),
                                           posterior_cs_attn.detach())
            kl_loss = cs_kl_loss + concept_kl_loss
        
            cs_enc = prior_cs_enc.squeeze(1)
            concept_enc = prior_concept_enc.squeeze(1)
        
            cs_fake_enc = torch.cat((cs_enc[-1].unsqueeze(0), cs_enc[:-1]), dim=0)
            concept_fake_enc = torch.cat((concept_enc[-1].unsqueeze(0), concept_enc[:-1]), dim=0)
            # mim_loss = self.infomax(cs_enc, cs_fake_enc, concept_enc, concept_fake_enc)
            coarse_mim_loss = self.infomax_score(cs_enc, cs_fake_enc, concept_enc, concept_fake_enc)
        
            # Fine-grained MIM
            bsz, react_uttr_num, _ = batch["react_batch"].size()
            assert uttr_num == react_uttr_num
            react_batch = batch["react_batch"].view(bsz*react_uttr_num, -1)
            react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
            react_emb = self.embedding(react_batch)
            react_batch_outputs = self.react_encoder(react_emb, react_batch_mask)
            # react_batch_enc, _ = self.react_selfattn(react_batch_outputs, react_batch_mask.squeeze(1))
            react_batch_enc = torch.mean(react_batch_outputs, dim=1)
            react_batch_enc = react_batch_enc.view(bsz, react_uttr_num, -1)
            react_batch_enc = self.react_ctx_encoder(torch.cat((react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                                                                enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)), 
                                                                dim=-1).view(bsz*react_uttr_num, enc_outputs.size(1), -1), 
                                                                src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz*react_uttr_num, -1,enc_outputs.size(1)))
            react_batch_enc = react_batch_enc[:,0,:].view(bsz, react_uttr_num, -1)
            react_batch_enc = self.react_linear(react_batch_enc)
            bsz, _, emb = react_batch_enc.size()
            uttr_emotion = react_batch_enc[:,0].unsqueeze(1).repeat(1, batch["max_uttr_cs_num"], 1)
            split_intent_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_intent_num"], 1).view(bsz, -1, emb)
            split_need_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_need_num"], 1).view(bsz, -1, emb)
            split_want_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_want_num"], 1).view(bsz, -1, emb)
            split_effect_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_effect_num"], 1).view(bsz, -1, emb)
            react_enc = torch.cat((uttr_emotion, 
                                   split_intent_emotion,
                                   split_need_emotion,
                                   split_want_emotion,
                                   split_effect_emotion), dim=1)
            assert react_enc.size(1) == cs_num
            react_fake_enc = torch.cat((react_enc[-1].unsqueeze(0), react_enc[:-1]), dim=0)
            commonsense_fake_outputs = torch.cat((commonsense_outputs[-1].unsqueeze(0), commonsense_outputs[:-1]), dim=0)
            fine_mim_loss = self.fine_grained_infomax_score(commonsense_outputs, commonsense_fake_outputs, react_enc, react_fake_enc, commonsense_mask)
        
            # === v2 Step2: 冻结后同步关闭 MIM，释放特征空间给 EPCL ===
            if train and is_currently_frozen:
                mim_loss = torch.tensor(0.0, device=config.device)
            else:
                mim_loss = config.coarse_weight * coarse_mim_loss + config.fine_weight * fine_mim_loss
        
            # uttr_split_react_mask = batch["uttr_split_react_mask"]
            # fine_mask = torch.cat((torch.ones((bsz, 1)).long().to(config.device), uttr_split_react_mask), dim=1)
            # assert fine_mask.size(1) == react_batch_enc.size(1)
            # fine_emotion, _ = self.fine_emotion_selfattn(react_batch_enc, fine_mask.eq(0))
        
            fine_emotion = react_batch_enc[:, 0]
            emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
            static_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
            
            # [MoP-DR] Prototype dynamic routing (compatible with K*C prototypes)
            if self.dataset == "ED":
                projected_features = self.epcl_criterion.projection_head(fine_emotion)
                if getattr(self.epcl_criterion, "use_erp", False):
                    proj_norm = projected_features
                else:
                    proj_norm = F.normalize(projected_features, p=2, dim=1)
                proto_norm = F.normalize(self.epcl_criterion.current_prototypes.detach(), p=2, dim=1)
                proto_logits = torch.matmul(proj_norm, proto_norm.T) / 0.1  # [B, K*C]
                
                P_route = F.softmax(proto_logits, dim=-1)  # [B, K*C]
                route_gate = torch.sigmoid(self.router_linear(P_route))
                current_lambda = config.lambda_epcl * (iter / config.epcl_warmup) if (iter < config.epcl_warmup and train) else config.lambda_epcl
                emo_gate = current_lambda * route_gate + (1 - current_lambda) * static_gate
            else:
                emo_gate = static_gate
            
            emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
        
            # [V8 手术 2 & 4] 特征挂载锚点选择 (Unified vs Decoupled Feature Anchors)
            if getattr(self, "epcl_anchor", "fine_emotion") == "emotion_enc":
                target_emotion_feat = emotion_enc
            else:
                target_emotion_feat = fine_emotion

            cls_anchor = getattr(self, "cls_anchor", getattr(self, "epcl_anchor", "fine_emotion"))
            if cls_anchor == "fine_emotion":
                cls_emotion_feat = fine_emotion
            elif cls_anchor == "emotion_enc":
                cls_emotion_feat = emotion_enc
            else:
                cls_emotion_feat = target_emotion_feat

            # 显式记录 target 表征投影 (供流形几何度量与可视化直接读取)
            if hasattr(self, "epcl_criterion") and hasattr(self.epcl_criterion, "projection_head"):
                with torch.no_grad():
                    proj_temp = self.epcl_criterion.projection_head(target_emotion_feat)
                    if getattr(self.epcl_criterion, "use_erp", False):
                        self.current_projected_emotion = proj_temp
                    else:
                        self.current_projected_emotion = F.normalize(proj_temp, p=2, dim=1)

            # === EPCL: 原型对比学习损失计算 ===
            # [v3 P3] Temperature Annealing (余弦退火: 0.3 -> 0.1, 回滚单锚点)
            tau_max = 0.3
            tau_min = 0.1
            max_epcl_step = 20000  # 对应 main.py 的 iters
            current_tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * min(iter, max_epcl_step) / max_epcl_step)) / 2

            # [v3 P3 / V7 / V8] 单锚点 EPCL 与 CCHP 位移正则项计算 (输入 target_emotion_feat)
            if self.dataset == "ED" and train:
                ctx_repr_for_cchp = enc_outputs[:, 0, :] if getattr(self.epcl_criterion, 'use_cchp', False) else None
                epcl_loss, cchp_reg_loss = self.epcl_criterion(target_emotion_feat, batch["program_label"], tau=current_tau, ctx_repr=ctx_repr_for_cchp)
            else:
                epcl_loss = torch.tensor(0.0, device=config.device)
                cchp_reg_loss = torch.tensor(0.0, device=config.device)

            # === EPCL: 分类头活性控制（冻结后关闭或软退火 CE 损失） ===
            emo_head_active = (not train) or (not is_currently_frozen)

            # === EPCL: λ_epcl 线性预热调度 ===
            if iter < config.epcl_warmup and train:
                lambda_epcl = config.lambda_epcl * (iter / config.epcl_warmup)
            else:
                lambda_epcl = config.lambda_epcl

            # emotion prediction（加入 Dropout 正则化）
            if self.dataset == "ED":
                # [V8 Trial 4] 分类头挂载至 cls_emotion_feat（支持与生成端双锚点解耦）
                emotion_logits = self.emotion_linear(self.emo_dropout(cls_emotion_feat))
                emotion_loss_raw = self.criterion_ce(emotion_logits, batch["program_label"])
                
                if emo_head_active:
                    emotion_loss = emotion_loss_raw
                else:
                    if getattr(config, 'soft_freeze', False):
                        # [V8 Trial 4] 软退火：以超小衰减权重持续优化，消除后半程特征漂移
                        decay_w = getattr(config, 'freeze_decay_weight', 0.05)
                        emotion_loss = decay_w * emotion_loss_raw
                    else:
                        # 传统硬冻结：损失置零
                        emotion_loss = torch.tensor(0.0, device=config.device)

                pred_emotion = np.argmax(emotion_logits.detach().cpu().numpy(), axis=1)
                emotion_acc = accuracy_score(batch["program_label"].detach().cpu().numpy(), pred_emotion)
        
            # Merge Context, Cognition-Affection-Strategy Signals
            if self.dataset == "ESConv":
                ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                    enc_outputs, 
                    cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                    concept_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                    strategy_enc_outputs.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
                ), dim=2))
            
                # predict next strategy
                strategy_logits = self.strategy_linear(ctx_enc_outputs[:,0,:])
                strategy_label = batch["strategy_label"]
                str_loss = self.criterion_ce(strategy_logits, strategy_label)
            
                pred_strategy = np.argmax(strategy_logits.detach().cpu().numpy(), axis=1)
                strategy_acc = accuracy_score(batch["strategy_label"].detach().cpu().numpy(), pred_strategy)
            else:
                ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                    enc_outputs, 
                    cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                    emotion_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                ), dim=2))
        
            # Decoder
            sos_token = (
                torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
                .unsqueeze(1)
                .to(config.device)
            )
            dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
            mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        
            # batch_size * seq_len * 300 (GloVe)
            dec_emb = self.embedding(dec_batch_shift)
            pre_logit, attn_dist = self.decoder(
                dec_emb, 
                ctx_enc_outputs, 
                (src_mask, mask_trg),
                cs_enc_outputs=commonsense_outputs,
                cs_enc_mask=commonsense_mask.eq(0).unsqueeze(1),
                concept_enc_outputs=src_concept_outputs,
                concept_enc_mask=src_concept_mask.unsqueeze(1),
            )
        
            # [V6/V7 PCAM] 原型交叉记忆注意力注入
            ctx_repr_for_pcam = enc_outputs[:, 0, :] if getattr(self.epcl_criterion, 'use_cchp', False) else None
            pre_logit = self.apply_pcam(pre_logit, ctx_repr=ctx_repr_for_pcam)

            ## compute output dist
            logit = self.generator(
                pre_logit,
                attn_dist,
                enc_batch_extend_vocab if config.pointer_gen else None,
                extra_zeros,
                attn_dist_db=None,
            )
        
            ctx_loss = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
        
            # reference CEM
            _, preds = logit.max(dim=-1)
            preds = self.clean_preds(preds)
            self.update_frequency(preds)
            self.criterion.weight = self.calc_weight()
            not_pad = dec_batch.ne(config.PAD_idx)
            target_tokens = not_pad.long().sum().item()
            div_loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
            div_loss /= target_tokens
        
            # [V4 Trial 8] Decoder MIM Loss (全周期软约束激活，修复分类头冻结后断开的 Bug)
            if self.dataset == "ED":
                mask_bool = mask_trg.squeeze(1) # [bsz, seq_len]
                valid_lens = (~mask_bool).sum(dim=-1, keepdim=True).float().clamp(min=1.0)
                pooled_dec = ((~mask_bool).unsqueeze(-1).float() * pre_logit).sum(dim=1) / valid_lens
                dec_emo_logits = self.emotion_linear(self.emo_dropout(pooled_dec))
                dec_emo_loss = self.criterion_ce(dec_emo_logits, batch["program_label"]) if train else torch.tensor(0.0, device=config.device)
            else:
                dec_emo_loss = torch.tensor(0.0, device=config.device)
        
            # [V7 Phase 2] 序列级无似然训练损失 (Unlikelihood Training)
            if getattr(config, 'use_unlikelihood', False) and train and self.dataset == "ED":
                ul_loss = compute_unlikelihood_loss(logit, dec_batch, pad_idx=config.PAD_idx)
            else:
                ul_loss = torch.tensor(0.0, device=config.device)

            if self.dataset == "ED":
                alpha_mim = config.alpha_mim  # [V5] 从命令行读取，默认 0.1（V4 Trial 8 基线值）
                # [V5 Trial 3] div_weight 可通过 --div_weight 调节（默认 2.0x，V4 Trial 8 基线值）
                cchp_reg_w = getattr(config, 'cchp_reg_weight', 0.01)
                ul_weight = getattr(config, 'unlikelihood_weight', 0.1)
                loss = (bow_loss + kl_loss + mim_loss + ctx_loss +
                        config.div_weight * div_loss + emotion_loss +
                        lambda_epcl * epcl_loss + alpha_mim * dec_emo_loss +
                        cchp_reg_w * cchp_reg_loss +
                        ul_weight * ul_loss)
            else:
                loss = bow_loss + kl_loss + mim_loss + ctx_loss + 1.5 * div_loss + str_loss
            
        if train:
            self.scaler.scale(loss).backward()
            if (iter + 1) % accum_steps == 0:
                self.scaler.step(self.optimizer.optimizer if config.noam else self.optimizer)
                self.scaler.update()
            
            # [V4 Trial 8 / V5 Trial 4] 冻结后 LR 衰减：从当前 LR (~3.125e-4) 衰减至 ~1e-5
            # 主训练阶段 scaler.step() 绕过了 NoamOpt，LR 恒定不变，此处手动干预衰减
            # [V5 Trial 1] 支持 linear(线性) / cosine(余弦退火) 两种衰减策略
            # [V5 Trial 4] 自适应冻结动态对齐退火起点 freeze_anchor
            freeze_anchor = self.actual_freeze_step if config.adaptive_freeze else config.epcl_freeze_step
            if config.noam and is_currently_frozen and iter > freeze_anchor:
                total_decay_steps = 22000.0  # 退火跨度（约 22k 步平滑衰减至 1e-5）
                progress = min((iter - freeze_anchor) / total_decay_steps, 1.0)
                base_lr = self.optimizer._rate if self.optimizer._rate > 0 else 3.125e-4
                lr_min = base_lr * 0.03  # 最低 LR ≈ 1e-5

                if config.lr_schedule == "cosine":
                    # 余弦退火：前期保持较高 LR 充分探索，后期快速精细收敛
                    decayed_lr = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress))
                else:
                    # 线性衰减（V4 Trial 8 默认行为）
                    decay_factor = 1.0 - 0.97 * progress
                    decayed_lr = base_lr * decay_factor

                for pg in self.optimizer.optimizer.param_groups:
                    pg["lr"] = decayed_lr
        
        return (
            bow_loss.item(),
            kl_loss.item(),
            mim_loss.item(),
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            str_loss.item() if self.dataset=="ESConv" else 0,
            strategy_acc.item() if self.dataset=="ESConv" else 0,
            emotion_loss.item() if self.dataset=="ED" else 0,
            emotion_acc.item() if self.dataset=="ED" else 0,
            epcl_loss.item() if self.dataset=="ED" else 0,
            dec_emo_loss.item() if self.dataset=="ED" else 0  # [V4 Trial 8] Decoder MIM 损失可观测性
        )
    
    def decoder_greedy(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_vad_batch = batch["context_vad"]
        
        # Encode Context
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        
        # Affection: Encode Concept
        concept_input = batch["concept_batch"]
        concept_mask = concept_input.data.eq(config.PAD_idx).unsqueeze(1)
        concept_vad_batch = batch["concept_vad_batch"]
        mask_concept = batch["mask_concept"]
        concept_adj_mask = batch["concept_adjacency_mask_batch"]
        # mask_concept = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
        # concept_mask = self.embedding(mask_concept)  # KG_idx embedding
        concept_emb = self.embedding(concept_input) + self.embedding(mask_concept)  # KG_idx embedding
        src_concept_input_emb = torch.cat((src_emb, concept_emb), dim=1)
        src_concept_outputs = self.concept_graph_encoder(src_concept_input_emb, 
                                                         src_concept_input_emb,
                                                         src_concept_input_emb,
                                                         concept_adj_mask)
        src_concept_mask = torch.cat((enc_batch, concept_input), dim=1).data.eq(config.PAD_idx)
        src_concept_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)
        src_concept_vad = torch.softmax(src_concept_vad, dim=-1).unsqueeze(2).repeat(1, 1, config.emb_dim)
        src_concept_outputs = self.vad_layernorm(src_concept_vad * src_concept_outputs)
        
        # Cognition: Encode Commonsense
        bsz, uttr_num, uttr_length = batch["uttr_batch_concat"].size()
        uttr_batch_concat = batch["uttr_batch_concat"].view(bsz*uttr_num, -1)
        uttr_batch_mask = uttr_batch_concat.data.eq(config.PAD_idx).unsqueeze(1)
        
        bsz, cs_num, cs_length = batch["cs_batch"].size()
        cs_batch = batch["cs_batch"].view(bsz*cs_num, -1)
        cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        cs_mask = batch["cs_mask"] # bsz, cs_num
        
        cs_adj_mask = batch["cs_adjacency_mask_batch"]
        
        uttr_batch_emb = self.embedding(uttr_batch_concat)
        uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_batch_mask)[:,0].view(bsz, uttr_num, -1)
        
        cs_batch_emb = self.embedding(cs_batch)
        cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)[:,0].view(bsz, cs_num, -1)
        uttr_cs_outputs = torch.cat((uttr_batch_outputs, cs_batch_outputs), dim=1)
        
        assert uttr_cs_outputs.size(1) == cs_adj_mask.size(1)
        relation_emb = self.relation_embedding(cs_adj_mask)
        uttr_cs_graph_outputs = self.cs_graph_encoder(uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  cs_adj_mask,
                                                  relation_emb)
        
        commonsense_outputs = uttr_cs_graph_outputs[:,-cs_batch_outputs.size(1):,:]
        commonsense_mask = cs_mask
        if self.dataset == "ESConv":
            # Strategy: Encode Strategy Sequence
            strategy_seqs = batch["strategy_seqs_batch"]
            mask_strategy = strategy_seqs.data.eq(config.PAD_idx).unsqueeze(1)
            strategy_seqs_emb = self.strategy_embedding(strategy_seqs)
            strategy_seqs_emb = self.add_position_embedding(strategy_seqs, strategy_seqs_emb)
            strategy_enc_outputs = self.strategy_encoder(strategy_seqs_emb, mask_strategy)
            strategy_enc_outputs = strategy_enc_outputs[:,0,:]
            
            prior_query = self.tanh(self.prior_query_linear(torch.cat((enc_outputs[:,0,:], strategy_enc_outputs), dim=-1)))
        else:
            prior_query = self.tanh(self.prior_query_linear(enc_outputs[:,0,:]))
        
        # concept
        prior_concept_enc, prior_concept_attn = self.concept_prior_attn(
            query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
            memory = self.tanh(src_concept_outputs),
            mask = src_concept_mask
        )
        prior_concept_attn = prior_concept_attn.squeeze(1)
        
        # commonsense
        prior_cs_enc, prior_cs_attn = self.cs_prior_attn(
            query = prior_query.unsqueeze(1), # enc_outputs[:,0,:].unsqueeze(1),
            memory = self.tanh(commonsense_outputs),
            mask = commonsense_mask.eq(0)
        )
        prior_cs_attn = prior_cs_attn.squeeze(1)
        cs_enc = prior_cs_enc.squeeze(1)
        concept_enc = prior_concept_enc.squeeze(1)
        
        # Fine-grained MIM
        bsz, react_uttr_num, _ = batch["react_batch"].size()
        assert uttr_num == react_uttr_num
        react_batch = batch["react_batch"].view(bsz*react_uttr_num, -1)
        react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
        react_emb = self.embedding(react_batch)
        react_batch_outputs = self.react_encoder(react_emb, react_batch_mask)
        # react_batch_enc, _ = self.react_selfattn(react_batch_outputs, react_batch_mask.squeeze(1))
        react_batch_enc = torch.mean(react_batch_outputs, dim=1)
        react_batch_enc = react_batch_enc.view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_ctx_encoder(torch.cat((react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                                                        enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)), 
                                                        dim=-1).view(bsz*react_uttr_num, enc_outputs.size(1), -1), 
                                                        src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz*react_uttr_num, -1,enc_outputs.size(1)))
        react_batch_enc = react_batch_enc[:,0,:].view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_linear(react_batch_enc)
        bsz, _, emb = react_batch_enc.size()
        uttr_emotion = react_batch_enc[:,0].unsqueeze(1).repeat(1, batch["max_uttr_cs_num"], 1)
        split_intent_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_intent_num"], 1).view(bsz, -1, emb)
        split_need_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_need_num"], 1).view(bsz, -1, emb)
        split_want_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_want_num"], 1).view(bsz, -1, emb)
        split_effect_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_effect_num"], 1).view(bsz, -1, emb)
        react_enc = torch.cat((uttr_emotion, 
                               split_intent_emotion,
                               split_need_emotion,
                               split_want_emotion,
                               split_effect_emotion), dim=1)
        assert react_enc.size(1) == cs_num
   
        # uttr_split_react_mask = batch["uttr_split_react_mask"]
        # fine_mask = torch.cat((torch.ones((bsz, 1)).long().to(config.device), uttr_split_react_mask), dim=1)
        # assert fine_mask.size(1) == react_batch_enc.size(1)
        # fine_emotion, _ = self.fine_emotion_selfattn(react_batch_enc, fine_mask.eq(0))
        
        fine_emotion = react_batch_enc[:, 0]
        emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
        static_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
        
        if self.dataset == "ED":
            projected_features = self.epcl_criterion.projection_head(fine_emotion)
            if getattr(self.epcl_criterion, "use_erp", False):
                proj_norm = projected_features
            else:
                proj_norm = F.normalize(projected_features, p=2, dim=1)
            proto_norm = F.normalize(self.epcl_criterion.current_prototypes, p=2, dim=1)
            proto_logits = torch.matmul(proj_norm, proto_norm.T) / 0.1  # [B, K*C]
            # [MoP-DR] Greedy decode routing (K*C compatible)
            P_route = F.softmax(proto_logits, dim=-1)  # [B, K*C]
            
            route_gate = torch.sigmoid(self.router_linear(P_route))
            emo_gate = config.lambda_epcl * route_gate + (1 - config.lambda_epcl) * static_gate
        else:
            emo_gate = static_gate
            
        emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
        
        # Merge Context, Cognition-Affection-Strategy Signals
        if self.dataset == "ESConv":
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                concept_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                strategy_enc_outputs.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
            ), dim=2))
        else:
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                emotion_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
            ), dim=2))

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_enc_outputs, (src_mask, mask_trg),
                    cs_enc_outputs=commonsense_outputs,
                    cs_enc_mask=commonsense_mask.eq(0).unsqueeze(1),
                    concept_enc_outputs=src_concept_outputs,
                    concept_enc_mask=src_concept_mask.unsqueeze(1),
                )

            # [V6/V7 PCAM] 原型交叉记忆注意力注入
            out = self.apply_pcam(out, ctx_repr=enc_outputs[:, 0, :] if getattr(self.epcl_criterion, 'use_cchp', False) else None)

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_sampling(self, batch, max_dec_step=30, temp=0.7, top_p=0.9, top_k=0):
        """
        [路线 A: 多样性自适应采样解码 (Nucleus / Top-p Sampling)]
        机制: 
        1. 完整复用 CASE 前端常识认知编码、概念情感图编码、MoP-DR 原型路由与 PCAM 原型交叉记忆强化；
        2. 自回归单步中，使用对数概率除以温度系数 temp，通过 top_k_top_p_filtering 截断尾部低概率噪词；
        3. 采用 torch.multinomial 随机多项式采样，彻底打破确定性 Beam Search 在尖锐分布下的死板套话循环。
        """
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        enc_vad_batch = batch["context_vad"]
        
        # 1. 编码对话上下文 (Context)
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        mask_emb = self.embedding(batch["mask_input"])
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)
        
        # 2. 情感流: 编码 ConceptNet 概念图 (Affection)
        concept_input = batch["concept_batch"]
        concept_mask = concept_input.data.eq(config.PAD_idx).unsqueeze(1)
        concept_vad_batch = batch["concept_vad_batch"]
        mask_concept = batch["mask_concept"]
        concept_adj_mask = batch["concept_adjacency_mask_batch"]
        concept_emb = self.embedding(concept_input) + self.embedding(mask_concept)
        src_concept_input_emb = torch.cat((src_emb, concept_emb), dim=1)
        src_concept_outputs = self.concept_graph_encoder(src_concept_input_emb, 
                                                         src_concept_input_emb,
                                                         src_concept_input_emb,
                                                         concept_adj_mask)
        src_concept_mask = torch.cat((enc_batch, concept_input), dim=1).data.eq(config.PAD_idx)
        src_concept_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)
        src_concept_vad = torch.softmax(src_concept_vad, dim=-1).unsqueeze(2).repeat(1, 1, config.emb_dim)
        src_concept_outputs = self.vad_layernorm(src_concept_vad * src_concept_outputs)
        
        # 3. 认知流: 编码 COMET 常识知识 (Cognition)
        bsz, uttr_num, uttr_length = batch["uttr_batch_concat"].size()
        uttr_batch_concat = batch["uttr_batch_concat"].view(bsz*uttr_num, -1)
        uttr_batch_mask = uttr_batch_concat.data.eq(config.PAD_idx).unsqueeze(1)
        
        bsz, cs_num, cs_length = batch["cs_batch"].size()
        cs_batch = batch["cs_batch"].view(bsz*cs_num, -1)
        cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        cs_mask = batch["cs_mask"]
        cs_adj_mask = batch["cs_adjacency_mask_batch"]
        
        uttr_batch_emb = self.embedding(uttr_batch_concat)
        uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_batch_mask)[:,0].view(bsz, uttr_num, -1)
        
        cs_batch_emb = self.embedding(cs_batch)
        cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)[:,0].view(bsz, cs_num, -1)
        uttr_cs_outputs = torch.cat((uttr_batch_outputs, cs_batch_outputs), dim=1)
        relation_emb = self.relation_embedding(cs_adj_mask)
        uttr_cs_graph_outputs = self.cs_graph_encoder(uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  uttr_cs_outputs,
                                                  cs_adj_mask,
                                                  relation_emb)
        
        commonsense_outputs = uttr_cs_graph_outputs[:,-cs_batch_outputs.size(1):,:]
        commonsense_mask = cs_mask
        if self.dataset == "ESConv":
            strategy_seqs = batch["strategy_seqs_batch"]
            mask_strategy = strategy_seqs.data.eq(config.PAD_idx).unsqueeze(1)
            strategy_seqs_emb = self.strategy_embedding(strategy_seqs)
            strategy_seqs_emb = self.add_position_embedding(strategy_seqs, strategy_seqs_emb)
            strategy_enc_outputs = self.strategy_encoder(strategy_seqs_emb, mask_strategy)
            strategy_enc_outputs = strategy_enc_outputs[:,0,:]
            prior_query = self.tanh(self.prior_query_linear(torch.cat((enc_outputs[:,0,:], strategy_enc_outputs), dim=-1)))
        else:
            prior_query = self.tanh(self.prior_query_linear(enc_outputs[:,0,:]))
        
        prior_concept_enc, prior_concept_attn = self.concept_prior_attn(
            query = prior_query.unsqueeze(1),
            memory = self.tanh(src_concept_outputs),
            mask = src_concept_mask
        )
        
        # commonsense
        prior_cs_enc, prior_cs_attn = self.cs_prior_attn(
            query = prior_query.unsqueeze(1),
            memory = self.tanh(commonsense_outputs),
            mask = commonsense_mask.eq(0)
        )
        prior_cs_attn = prior_cs_attn.squeeze(1)
        cs_enc = prior_cs_enc.squeeze(1)
        concept_enc = prior_concept_enc.squeeze(1)
        
        # Fine-grained MIM
        bsz, react_uttr_num, _ = batch["react_batch"].size()
        assert uttr_num == react_uttr_num
        react_batch = batch["react_batch"].view(bsz*react_uttr_num, -1)
        react_batch_mask = react_batch.data.eq(config.PAD_idx).unsqueeze(1)
        react_emb = self.embedding(react_batch)
        react_batch_outputs = self.react_encoder(react_emb, react_batch_mask)
        react_batch_enc = torch.mean(react_batch_outputs, dim=1)
        react_batch_enc = react_batch_enc.view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_ctx_encoder(torch.cat((react_batch_enc.unsqueeze(2).repeat(1, 1, enc_outputs.size(1), 1),
                                                        enc_outputs.unsqueeze(1).repeat(1, react_uttr_num, 1, 1)), 
                                                        dim=-1).view(bsz*react_uttr_num, enc_outputs.size(1), -1), 
                                                        src_mask.unsqueeze(1).repeat(1, react_uttr_num, 1, 1).view(bsz*react_uttr_num, -1,enc_outputs.size(1)))
        react_batch_enc = react_batch_enc[:,0,:].view(bsz, react_uttr_num, -1)
        react_batch_enc = self.react_linear(react_batch_enc)
        bsz, _, emb = react_batch_enc.size()
        uttr_emotion = react_batch_enc[:,0].unsqueeze(1).repeat(1, batch["max_uttr_cs_num"], 1)
        split_intent_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_intent_num"], 1).view(bsz, -1, emb)
        split_need_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_need_num"], 1).view(bsz, -1, emb)
        split_want_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_want_num"], 1).view(bsz, -1, emb)
        split_effect_emotion = react_batch_enc[:,1:].unsqueeze(2).repeat(1, 1, batch["split_effect_num"], 1).view(bsz, -1, emb)
        react_enc = torch.cat((uttr_emotion, 
                               split_intent_emotion,
                               split_need_emotion,
                               split_want_emotion,
                               split_effect_emotion), dim=1)
        assert react_enc.size(1) == cs_num
        fine_emotion = react_batch_enc[:, 0]
        emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
        static_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
        
        # 4. 原型路由 (MoP-DR)
        if self.dataset == "ED":
            projected_features = self.epcl_criterion.projection_head(fine_emotion)
            if getattr(self.epcl_criterion, "use_erp", False):
                proj_norm = projected_features
            else:
                proj_norm = F.normalize(projected_features, p=2, dim=1)
            proto_norm = F.normalize(self.epcl_criterion.current_prototypes, p=2, dim=1)
            proto_logits = torch.matmul(proj_norm, proto_norm.T) / 0.1
            P_route = F.softmax(proto_logits, dim=-1)
            route_gate = torch.sigmoid(self.router_linear(P_route))
            emo_gate = config.lambda_epcl * route_gate + (1 - config.lambda_epcl) * static_gate
        else:
            emo_gate = static_gate
            
        emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
        
        # 5. 上下文融合 (Context Merge)
        if self.dataset == "ESConv":
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                concept_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                strategy_enc_outputs.unsqueeze(1).repeat(1, enc_outputs.size(1), 1)
            ), dim=2))
        else:
            ctx_enc_outputs = self.ctx_merge_lin(torch.cat((
                enc_outputs, 
                cs_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
                emotion_enc.unsqueeze(1).repeat(1, enc_outputs.size(1), 1),
            ), dim=2))

        # 6. 自回归采样生成循环
        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_enc_outputs, (src_mask, mask_trg),
                    cs_enc_outputs=commonsense_outputs,
                    cs_enc_mask=commonsense_mask.eq(0).unsqueeze(1),
                    concept_enc_outputs=src_concept_outputs,
                    concept_enc_mask=src_concept_mask.unsqueeze(1),
                )

            # [V6/V7 PCAM] 原型交叉记忆注意力注入
            out = self.apply_pcam(out, ctx_repr=enc_outputs[:, 0, :] if getattr(self.epcl_criterion, 'use_cchp', False) else None)

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            
            logits_step = prob[:, -1]  # [1, vocab_size] (log probabilities)
            if temp > 0:
                scaled_logits = logits_step / max(temp, 1e-4)
                filtered_logits = top_k_top_p_filtering(scaled_logits.squeeze(0), top_k=top_k, top_p=top_p)
                sample_probs = F.softmax(filtered_logits, dim=-1)
                next_word = torch.multinomial(sample_probs, 1)
            else:
                _, next_word = torch.max(logits_step, dim=-1)
                next_word = next_word.unsqueeze(0)

            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word_idx = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word_idx).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

        # Cognition: Encode Commonsense
        # bsz = batch["uttr_batch"].size(0)
        # uttr_batch = batch["uttr_batch"]
        # uttr_mask = uttr_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # bsz, uttr_split_num, _ = batch["uttr_split_batch"].size()
        # uttr_split_batch = batch["uttr_split_batch"].view(bsz*uttr_split_num, -1)
        # uttr_split_mask = uttr_split_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # uttr_batch_emb = self.embedding(uttr_batch)
        # uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_mask)[:,0].unsqueeze(1)
        # uttr_split_batch_emb = self.embedding(uttr_split_batch)
        # uttr_split_batch_ouptuts = self.cognition_encoder(uttr_split_batch_emb, uttr_split_mask)[:,0].view(bsz, uttr_split_num, -1)
        
        # bsz, cs_num, _ = batch["x_intent_batch"].size()
        # cs_batch = torch.cat((batch["x_intent_batch"].view(bsz*cs_num, -1),
        #                       batch["x_need_batch"].view(bsz*cs_num, -1),
        #                       batch["x_want_batch"].view(bsz*cs_num, -1),
        #                       batch["x_effect_batch"].view(bsz*cs_num, -1),
        #                       ), dim=0)
        # cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # cs_mask = torch.cat((batch["x_intent_mask"],
        #                      batch["x_need_mask"],
        #                      batch["x_want_mask"],
        #                      batch["x_effect_mask"]), dim=1)
        
        # bsz, split_num, cs_split_num, _ = batch["x_intent_split_batch"].size()
        # cs_split_batch = torch.cat((batch["x_intent_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_need_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_want_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_effect_split_batch"].view(bsz*split_num*cs_split_num, -1)), dim=0)
        # cs_split_batch_mask = cs_split_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # cs_split_mask = torch.cat((batch["x_intent_split_mask"].view(bsz, -1),
        #                            batch["x_need_split_mask"].view(bsz, -1),
        #                            batch["x_want_split_mask"].view(bsz, -1),
        #                            batch["x_effect_split_mask"].view(bsz, -1)))
        
        # cs_batch_emb = self.embedding(cs_batch)
        # cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)
        # cs_split_batch_emb = self.embedding(cs_split_batch)
        # cs_split_batch_outputs = self.cognition_encoder(cs_split_batch_emb, cs_split_batch_mask)
        # cs_outputs = torch.cat((cs_batch_outputs[:,0].view(bsz, cs_num, -1), 
        #                         cs_split_batch_outputs[:,0].view(bsz, split_num*cs_split_num, -1)), dim=1)
        # src_cs_emb = torch.cat((uttr_batch_outputs, 
                                # uttr_split_batch_ouptuts,
                                # cs_outputs), dim=1)
        
        # cs_enc = torch.bmm(prior_cs_attn*commonsense_mask, commonsense_outputs)
        # src_concept_vad = torch.softmax(src_concept_vad, dim=-1)

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

        # Cognition: Encode Commonsense
        # bsz = batch["uttr_batch"].size(0)
        # uttr_batch = batch["uttr_batch"]
        # uttr_mask = uttr_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # bsz, uttr_split_num, _ = batch["uttr_split_batch"].size()
        # uttr_split_batch = batch["uttr_split_batch"].view(bsz*uttr_split_num, -1)
        # uttr_split_mask = uttr_split_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # uttr_batch_emb = self.embedding(uttr_batch)
        # uttr_batch_outputs = self.cognition_encoder(uttr_batch_emb, uttr_mask)[:,0].unsqueeze(1)
        # uttr_split_batch_emb = self.embedding(uttr_split_batch)
        # uttr_split_batch_ouptuts = self.cognition_encoder(uttr_split_batch_emb, uttr_split_mask)[:,0].view(bsz, uttr_split_num, -1)
        
        # bsz, cs_num, _ = batch["x_intent_batch"].size()
        # cs_batch = torch.cat((batch["x_intent_batch"].view(bsz*cs_num, -1),
        #                       batch["x_need_batch"].view(bsz*cs_num, -1),
        #                       batch["x_want_batch"].view(bsz*cs_num, -1),
        #                       batch["x_effect_batch"].view(bsz*cs_num, -1),
        #                       ), dim=0)
        # cs_batch_mask = cs_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # cs_mask = torch.cat((batch["x_intent_mask"],
        #                      batch["x_need_mask"],
        #                      batch["x_want_mask"],
        #                      batch["x_effect_mask"]), dim=1)
        
        # bsz, split_num, cs_split_num, _ = batch["x_intent_split_batch"].size()
        # cs_split_batch = torch.cat((batch["x_intent_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_need_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_want_split_batch"].view(bsz*split_num*cs_split_num, -1),
        #                       batch["x_effect_split_batch"].view(bsz*split_num*cs_split_num, -1)), dim=0)
        # cs_split_batch_mask = cs_split_batch.data.eq(config.PAD_idx).unsqueeze(1)
        # cs_split_mask = torch.cat((batch["x_intent_split_mask"].view(bsz, -1),
        #                            batch["x_need_split_mask"].view(bsz, -1),
        #                            batch["x_want_split_mask"].view(bsz, -1),
        #                            batch["x_effect_split_mask"].view(bsz, -1)))
        
        # cs_batch_emb = self.embedding(cs_batch)
        # cs_batch_outputs = self.cognition_encoder(cs_batch_emb, cs_batch_mask)
        # cs_split_batch_emb = self.embedding(cs_split_batch)
        # cs_split_batch_outputs = self.cognition_encoder(cs_split_batch_emb, cs_split_batch_mask)
        # cs_outputs = torch.cat((cs_batch_outputs[:,0].view(bsz, cs_num, -1), 
        #                         cs_split_batch_outputs[:,0].view(bsz, split_num*cs_split_num, -1)), dim=1)
        # src_cs_emb = torch.cat((uttr_batch_outputs, 
                                # uttr_split_batch_ouptuts,
                                # cs_outputs), dim=1)
        
        # cs_enc = torch.bmm(prior_cs_attn*commonsense_mask, commonsense_outputs)
        # src_concept_vad = torch.softmax(src_concept_vad, dim=-1)
        # concept_enc = torch.bmm(prior_concept_attn*src_concept_vad, src_concept_outputs)