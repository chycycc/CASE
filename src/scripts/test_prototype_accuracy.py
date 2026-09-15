import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq

def evaluate_accuracy_modes(checkpoint_path, epcl_anchor="emotion_enc", cls_anchor="default", use_arc_margin=True, arc_margin=0.30):
    config.batch_size = 16
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.test = True
    config.model = "case"
    config.beam_size = 5
    config.use_mcp = True
    config.mcp_momentum = 0.99
    config.use_erp = True
    config.erp_hidden_dim = 768
    config.erp_dropout = 0.1
    config.use_pcam = True
    config.pcam_heads = 2
    config.pcam_dropout = 0.1
    config.pcam_gate_bias = -1.0
    config.use_unlikelihood = True
    config.unlikelihood_weight = 0.1
    config.use_arc_margin = use_arc_margin
    config.arc_margin = arc_margin
    config.epcl_anchor = epcl_anchor
    config.cls_anchor = cls_anchor
    config.emotion_head_type = "residual_mlp"
    config.mlp_hidden_dim = 300
    config.mlp_dropout = 0.1
    config.seed = 13
    set_seed()

    print(f"[*] 加载数据...")
    _, _, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=config.batch_size)

    model = CASE(vocab, emotion_num=emo_num, strategy_num=strategy_num, is_eval=True)
    model.to(config.device)
    
    print(f"[*] 加载权重: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    all_labels = []
    mlp_acc_list = []
    proto_preds = []
    all_z = []

    # 原型矩阵
    proto_norm = F.normalize(model.epcl_criterion.prototypes.data, p=2, dim=1) # [32, 300]

    with torch.no_grad():
        for batch in tqdm(test_set, desc="Evaluating Test Set"):
            bow, kl, mim, ctx, ppl, str_l, str_a, emo_loss, emo_acc, epcl, dec_emo = model.train_one_batch(
                batch, 0, train=False
            )
            mlp_acc_list.append(emo_acc)
            
            # 获取当前 batch 的投影表征
            z = model.current_projected_emotion # [B, 300]
            labels = batch["program_label"].cpu().numpy()
            all_labels.extend(labels)

            # 原型分类预测: argmax(z @ proto^T)
            proto_sim = torch.matmul(z, proto_norm.T) # [B, 32]
            proto_pred = proto_sim.argmax(dim=-1).cpu().numpy()
            proto_preds.extend(proto_pred)

    acc_mlp = np.mean(mlp_acc_list) * 100
    acc_proto = accuracy_score(all_labels, proto_preds) * 100

    print(f"\n=================================================")
    print(f"  测试集情感分类准确率实测对比 (N={len(all_labels)})")
    print(f"=================================================")
    print(f"  [1] 传统 MLP 分类头准确率 (模型输出):  {acc_mlp:.2f}%")
    print(f"  [2] MCP 最近质心原型分类器 (NPC):      {acc_proto:.2f}%")
    print(f"=================================================")
    return acc_mlp, acc_proto

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "save/epcl_v8_trial4/CASE_51999_37.3874"
    anchor = sys.argv[2] if len(sys.argv) > 2 else ("fine_emotion" if "trial1" in ckpt else "emotion_enc")
    cls_anchor = sys.argv[3] if len(sys.argv) > 3 else ("fine_emotion" if "trial4" in ckpt else "default")
    evaluate_accuracy_modes(ckpt, epcl_anchor=anchor, cls_anchor=cls_anchor)

