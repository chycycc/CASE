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
    cuda_id = "cuda:" + str(args.gpu)
    args.device = torch.device(cuda_id) if torch.cuda.is_available() else 'cpu'
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
