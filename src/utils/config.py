import torch
import logging
import argparse


UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6

def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ED")
    parser.add_argument("--comet_file", type=str, default="data/Comet")
    parser.add_argument("--max_num_dialog", type=int, default=9)
    parser.add_argument("--concept_num", type=int, default=3)
    parser.add_argument("--total_concept_num", type=int, default=10, help='the maximum number of external concepts injection for a sentence.')
    parser.add_argument("--cs_num", type=int, default=5)
    parser.add_argument("--emb_file", type=str)
    parser.add_argument("--save_path", type=str, default="save/test")
    parser.add_argument("--model_path", type=str, default="save/test")
    parser.add_argument("--save_path_dataset", type=str, default="save/")

    # Model
    parser.add_argument("--UNK_idx", type=int, default=0)
    parser.add_argument("--PAD_idx", type=int, default=1)
    parser.add_argument("--EOS_idx", type=int, default=2)
    parser.add_argument("--SOS_idx", type=int, default=3)
    parser.add_argument("--USR_idx", type=int, default=4)
    parser.add_argument("--SYS_idx", type=int, default=5)
    parser.add_argument("--CLS_idx", type=int, default=6)
    parser.add_argument("--KG_idx", type=int, default=7)
    parser.add_argument("--SEP_idx", type=int, default=8)
    
    # cs relation
    parser.add_argument("--self_loop", type=int, default=2)
    parser.add_argument("--contain", type=int, default=3)
    parser.add_argument("--temporary", type=int, default=4)
    parser.add_argument("--intent_idx", type=int, default=5)
    parser.add_argument("--need_idx", type=int, default=6)
    parser.add_argument("--want_idx", type=int, default=7)
    parser.add_argument("--effect_idx", type=int, default=8)
    parser.add_argument("--react_idx", type=int, default=9)
    parser.add_argument("--relation_num", type=int, default=10)
    
    # Train/Test
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="[V9 解绑妥协] 梯度累加步数 (24GB 大 Batch 推荐 1，4GB 显存调试时可设为 4)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                        help="[V9 解绑妥协] 计算精度 (24GB 推荐 fp32 全精度或 bf16，4GB 推荐 fp16)")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--beam_size", type=int, default=5)
    
    parser.add_argument("--pointer_gen", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--basic_learner", default=True, action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--softmax", default=True, action="store_true")
    parser.add_argument("--mean_query", action="store_true")
    parser.add_argument("--schedule", type=float, default=10000)
    
    ## transformer
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--filter", type=int, default=300)
    
    ## GraphTransformer
    parser.add_argument("--graph_layer_num", type=int, default=1)
    parser.add_argument("--graph_ffn_emb_dim", type=int, default=300)
    parser.add_argument("--graph_num_heads", type=int, default=2)
    
    # Other
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model", type=str, default="case")
    parser.add_argument("--cuda", default=True, action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split_data_seed", type=int, default=13)
    
    parser.add_argument("--pretrain", default=True, action="store_true")
    parser.add_argument("--pretrain_epoch", type=int, default=4)
    parser.add_argument("--woStrategy", default=True, action="store_true")
    parser.add_argument("--model_file_path", type=str, default="save/test/")
    parser.add_argument("--warmup", type=int, default=12000)
    parser.add_argument("--fine_weight", type=float, default=0.2)
    parser.add_argument("--coarse_weight", type=float, default=1.0)
    
    # EPCL 超参数（从 CEM-EPCL v6.4 移植的验证值）
    parser.add_argument("--lambda_epcl", type=float, default=0.07,
                        help="EPCL 对比损失权重")
    parser.add_argument("--epcl_warmup", type=int, default=3000,
                        help="EPCL λ 线性预热步数")
    parser.add_argument("--epcl_freeze_step", type=int, default=14000,
                        help="分类头冻结时间点")
    parser.add_argument("--alpha_mim", type=float, default=0.1,
                        help="Decoder MIM 损失权重（V4 Trial 8 基线=0.1）")
    parser.add_argument("--div_weight", type=float, default=2.0,
                        help="多样性损失 div_loss 权重系数（V4 Trial 8 基线=2.0）")
    parser.add_argument("--lr_schedule", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="冻结后 LR 衰减策略: linear(线性) / cosine(余弦退火)")
    
    # V5 Trial 4: 自适应分类头冻结机制 (Adaptive Classifier Freezing, ACF)
    parser.add_argument("--adaptive_freeze", action="store_true", default=False,
                        help="是否启用基于验证集收敛平台期的数据驱动自适应分类头冻结机制")
    parser.add_argument("--freeze_patience", type=int, default=3,
                        help="验证集指标未改善的容忍次数（默认3次，每2000步一次共6000步）")
    parser.add_argument("--min_freeze_step", type=int, default=16000,
                        help="自适应冻结评估的最早步数下界（防止冷启动欠拟合）")
    parser.add_argument("--max_freeze_step", type=int, default=32000,
                        help="自适应冻结评估的最晚兜底步数上界（防范极端情况未触发）")
    parser.add_argument("--freeze_metric", type=str, default="emo_acc",
                        choices=["emo_acc", "emo_loss"],
                        help="自适应冻结监控指标: emo_acc(准确率提升) / emo_loss(损失下降)")
    parser.add_argument("--rollback_best_freeze", action="store_true", default=False,
                        help="自适应冻结触发时是否将分类头回滚至验证集历史最高泛化状态 (BCF)")
    
    # V6 Trial 1: 双层非线性残差分类头 (Residual Emotion Head)
    parser.add_argument("--emotion_head_type", type=str, default="linear",
                        choices=["linear", "residual_mlp"],
                        help="情感分类头类型: linear(原单层线性) / residual_mlp(V6双层非线性残差头)")
    parser.add_argument("--mlp_hidden_dim", type=int, default=300,
                        help="残差分类头隐藏层维度 (默认300)")
    parser.add_argument("--mlp_dropout", type=float, default=0.1,
                        help="残差分类头 Dropout 比率 (默认0.1)")
    
    # V6 Trial 2: 类内多原型解耦 (Multi-Prototype EPCL)
    parser.add_argument("--num_prototypes_per_class", type=int, default=1,
                        help="每个情感类别的子原型数量 (默认1为单原型，V6 Trial 2设为2，共64个原型)")
    parser.add_argument("--alpha_uni", type=float, default=1.0,
                        help="EPCL 均匀性损失权重 (默认1.0, Trial 2b设为0.3以放松子原型排斥约束)")
    
    # V6 Trial 3: 原型交叉记忆注意力 (PCAM)
    parser.add_argument("--use_pcam", action="store_true", default=False,
                        help="是否在解码器输出端启用原型交叉记忆注意力模块 (PCAM)")
    parser.add_argument("--pcam_heads", type=int, default=2,
                        help="PCAM 交叉注意力的多头数量 (默认2)")
    parser.add_argument("--pcam_dropout", type=float, default=0.1,
                        help="PCAM 交叉注意力的 Dropout 概率 (默认0.1)")
    parser.add_argument("--pcam_gate_bias", type=float, default=-1.0,
                        help="PCAM 自适应门控的初始偏置 (默认-1.0，实现平滑暖启动)")
    
    # V7 架构超参数: 高维解缠残差投影头 (ERP) 与动态超网络原型 (CCHP)
    parser.add_argument("--use_erp", action="store_true", default=False,
                        help="[V7] 是否启用高维解缠残差投影头 (Expanded Residual Projector)")
    parser.add_argument("--erp_hidden_dim", type=int, default=768,
                        help="[V7] ERP 高维解缠扩张隐藏层维度 (默认768)")
    parser.add_argument("--erp_dropout", type=float, default=0.1,
                        help="[V7] ERP 投影头 Dropout 比率")
    parser.add_argument("--use_cchp", action="store_true", default=False,
                        help="[V7] 是否启用动态上下文条件超网络原型 (Context-Conditioned Hyper-Prototypes)")
    parser.add_argument("--cchp_alpha", type=float, default=0.1,
                        help="[V7] CCHP 语境自适应残差位移幅度系数 (默认0.1)")
    parser.add_argument("--cchp_reg_weight", type=float, default=0.01,
                        help="[V7] CCHP 动态位移 L2 正则化惩罚权重 (防止平凡解与表征崩塌)")
    parser.add_argument("--use_unlikelihood", action="store_true", default=False,
                        help="[V7] 是否在训练端启用序列级无似然训练损失 (Unlikelihood Training)")
    parser.add_argument("--unlikelihood_weight", type=float, default=0.1,
                        help="[V7 / V8.2 C] 序列级无似然训练损失权重 (默认0.1，V8.2 C 推荐 0.05)")
    # V8 架构超参数: 动量样本中心原型 (MCP) 与加性角度硬边际 (Arc-EPCL)
    parser.add_argument("--use_mcp", action="store_true", default=False,
                        help="[V8] 是否启用动量样本中心原型 (Momentum Centroid Prototypes)")
    parser.add_argument("--mcp_momentum", type=float, default=0.99,
                        help="[V8] MCP 样本质心指数移动平均动量系数 (默认0.99)")
    parser.add_argument("--use_arc_margin", action="store_true", default=False,
                        help="[V8] 是否启用加性角度硬边际对比损失 (Arc-EPCL)")
    parser.add_argument("--arc_margin", type=float, default=0.30,
                        help="[V8] Arc-EPCL 加性角度硬边际 m (默认0.30)")
    parser.add_argument("--arc_mode", type=str, default="cos", choices=["cos", "angle"],
                        help="[V8] Arc-EPCL 硬边际计算模式 (cos: cos(θ)-m, angle: cos(θ+m))")
    parser.add_argument("--epcl_anchor", type=str, default="fine_emotion", choices=["fine_emotion", "emotion_enc"],
                        help="[V8] EPCL 对比学习特征挂载锚点 (默认 fine_emotion，支持统一挂载至 emotion_enc)")
    parser.add_argument("--cls_anchor", type=str, default="default", choices=["default", "fine_emotion", "emotion_enc"],
                        help="[V8 Trial 4] 情感分类头专属特征锚点 (default: 跟随 epcl_anchor, fine_emotion: 纯净情绪锚点, emotion_enc: 融合锚点)")
    parser.add_argument("--soft_freeze", action="store_true", default=False,
                        help="[V8 Trial 4] 是否启用分类头时序软退火 (Soft Freeze, 替代彻底置死为0的硬冻结)")
    parser.add_argument("--freeze_decay_weight", type=float, default=0.05,
                        help="[V8 Trial 4] 软退火期间分类损失衰减权重系数 (默认0.05)")
    
    # V8.1 双轨实验超参数: 词表层情感偏置注入 (路线 A) 与时序余弦软退火/复合早停 (路线 B)
    parser.add_argument("--use_emo_bias", action="store_true", default=False,
                        help="[V8.1 路线 A] 是否启用基于 KEMP 的词表层显式情感偏置注入 (Emotion-Aware Vocab Bias)")
    parser.add_argument("--emo_bias_gate_init", type=float, default=-2.0,
                        help="[V8.1 路线 A] 词表层情感偏置门控标量初始值 (默认 -2.0，对应 sigmoid(-2.0) 约 0.119)")
    parser.add_argument("--disable_freeze", action="store_true", default=False,
                        help="[V8.1 路线 A] 显式禁用任何阶段分类头冻结，保持全程联合微调 (requires_grad=True)")
    parser.add_argument("--use_cosine_anneal", action="store_true", default=False,
                        help="[V8.1 路线 B] 是否启用多任务辅助损失时序余弦平滑软退火")
    parser.add_argument("--anneal_start_step", type=int, default=24000,
                        help="[V8.1 路线 B] 余弦平滑退火起始步数 (默认 24000)")
    parser.add_argument("--anneal_steps", type=int, default=16000,
                        help="[V8.1 路线 B] 余弦平滑退火持续步长 (默认 16000，即 24k~40k 步平滑衰减)")
    parser.add_argument("--anneal_min_weight", type=float, default=0.05,
                        help="[V8.1 路线 B] 余弦退火最低下限权重 (默认 0.05，保持极微弱微调更新)")
    # V8.2 破局架构方案超参数: 自适应拓扑稀疏掩码、PCGrad 梯度正交投影与动态门控预热
    parser.add_argument("--use_sparse_emo_bias", action="store_true", default=False,
                        help="[V8.2] 是否启用基于原型-词嵌入拓扑亲和度的自适应稀疏情感词表偏置")
    parser.add_argument("--emo_vocab_topk_ratio", type=float, default=0.15,
                        help="[V8.2] 自适应稀疏偏置保留的情感亲和词比例 (默认 0.15，即保留前 15% 情感词，85% 功能词零偏置)")
    parser.add_argument("--use_pcgrad", action="store_true", default=False,
                        help="[V8.2] 是否启用多任务冲突梯度正交投影器 (PCGrad)")
    parser.add_argument("--gate_warmup_steps", type=int, default=20000,
                        help="[V8.2] 情感偏置门控时序预热步数 (默认 20000，前 20k 步门控置冷锁定保护语言模型冷启动)")
    parser.add_argument("--mask_update_interval", type=int, default=5000,
                        help="[V8.2] 原型-词向量拓扑亲和度掩码动态刷新步频 (默认 5000 步更新一次)")
    parser.add_argument("--patience", type=int, default=5,
                        help="[V8.2 B] 早停最大容忍次数 (默认 5 次，即 10,000 步；V8.2 B 建议设为 12 次以保证退火完整走完)")
    # V8.2 C 参数: 偏置门控时序余弦退火超参数
    parser.add_argument("--use_bias_annealing", action="store_true", default=False,
                        help="[V8.2 C] 是否启用解码端偏置门控时序余弦退火衰减 (后程平滑降噪)")
    parser.add_argument("--bias_anneal_start", type=int, default=35000,
                        help="[V8.2 C] 偏置门控余弦退火起始步数 (默认 35000)")
    parser.add_argument("--bias_anneal_steps", type=int, default=15000,
                        help="[V8.2 C] 偏置门控余弦退火衰减持续步长 (默认 15000)")
    parser.add_argument("--bias_min_scale", type=float, default=0.2,
                        help="[V8.2 C] 偏置门控衰减下限乘子 (默认 0.2，保留 20% 偏置防重复)")
    
    # V8.2 D 参数: 后半程动态情感损失权重线性提升与帕累托复合检查点遴选
    parser.add_argument("--use_emo_loss_ramp", action="store_true", default=False,
                        help="[V8.2 D] 是否启用后半程动态情感损失权重线性提升 (抵御生成任务梯度支配)")
    parser.add_argument("--emo_loss_ramp_start", type=int, default=24000,
                        help="[V8.2 D] 动态情感损失权重提升起始步数 (默认 24000)")
    parser.add_argument("--emo_loss_ramp_steps", type=int, default=16000,
                        help="[V8.2 D] 动态情感损失权重提升跨度步数 (默认 16000，24k~40k)")
    parser.add_argument("--emo_loss_ramp_max", type=float, default=1.5,
                        help="[V8.2 D] 动态情感损失权重提升上限乘子 (默认 1.5)")
    parser.add_argument("--emo_loss_ramp_shape", type=str, default="linear",
                        choices=["linear", "bell"],
                        help="[V8.2 F] 动态情感损失权重变化形状 (linear: 单调递增, bell: 钟形余弦先升后降)")
    parser.add_argument("--use_composite_score", action="store_true", default=False,
                        help="[V8.1/V8.2 D] 是否启用验证集复合评分遴选检查点")
    parser.add_argument("--composite_acc_weight", type=float, default=20.0,
                        help="[V8.1] 复合评分中 EMO_acc 权衡权重 (默认 20.0)")
    parser.add_argument("--composite_mode", type=str, default="acc", choices=["acc", "emo_loss"],
                        help="[V8.2 D] 复合早停/评分模式 (acc: 传统 PPL-acc, emo_loss: 帕累托 PPL+alpha*EMO_loss)")
    parser.add_argument("--composite_emo_loss_weight", type=float, default=3.5,
                        help="[V8.2 D/E] 帕累托复合评分中 EMO_loss 权衡系数 alpha (默认 3.5)")
    parser.add_argument("--min_save_step", type=int, default=0,
                        help="[V8.2 E] 退火成熟保护期步数 (默认0，V8.2 E 推荐 32000，保护期内不累加早停耐心)")
    
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--large_decoder", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--is_coverage", action="store_true")
    parser.add_argument("--use_oov_emb", action="store_true")
    parser.add_argument("--pretrain_emb", default=True, action="store_true")
    parser.add_argument("--weight_sharing", action="store_true")
    parser.add_argument("--label_smoothing", default=True, action="store_true")
    parser.add_argument("--noam", default=True, action="store_true")
    parser.add_argument("--universal", action="store_true")
    parser.add_argument("--act", action="store_true")
    parser.add_argument("--act_loss_weight", type=float, default=0.001)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1, help='dropout')
    
    args, _ = parser.parse_known_args()
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        target_gpu = args.gpu
        if target_gpu < 0 or target_gpu >= gpu_count:
            logging.warning("[*] 指定的 GPU 编号 %s 超出系统可用范围 [0, %d)，已自动回退至 cuda:0", target_gpu, gpu_count)
            target_gpu = 0
            args.gpu = 0
        args.device = torch.device(f"cuda:{target_gpu}")
    else:
        args.device = torch.device("cpu")
    args.emb_file = args.emb_file or "vectors/glove.6B.{}d.txt".format(str(args.emb_dim))
    print_opts(args)
    
    return args

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if key == "device":
            continue
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)

config = get_args()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M"
)
collect_stats = False
