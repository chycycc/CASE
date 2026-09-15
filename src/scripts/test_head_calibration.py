import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq

def test_head_calibration(checkpoint_path, epcl_anchor="emotion_enc", calib_steps=1000, lr=1e-3):
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
    config.use_arc_margin = True
    config.arc_margin = 0.30
    config.epcl_anchor = epcl_anchor
    config.emotion_head_type = "residual_mlp"
    config.mlp_hidden_dim = 300
    config.mlp_dropout = 0.1
    config.seed = 13
    set_seed()

    print(f"[*] 加载训练集、验证集与测试集...")
    train_set, dev_set, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=config.batch_size)

    model = CASE(vocab, emotion_num=emo_num, strategy_num=strategy_num, is_eval=True)
    model.to(config.device)
    
    print(f"[*] 加载基座权重: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=config.device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    # 1. 评估校准前的测试集准确率
    model.eval()
    all_labels_before, mlp_acc_before = [], []
    with torch.no_grad():
        for batch in tqdm(test_set, desc="Pre-Calibration Test Eval"):
            _, _, _, _, _, _, _, _, emo_acc, _, _ = model.train_one_batch(batch, 0, train=False)
            mlp_acc_before.append(emo_acc)
    acc_before = np.mean(mlp_acc_before) * 100
    print(f"[+] 校准前原始测试集分类准确率: {acc_before:.2f}%")

    # 2. 快速提取训练集、验证集与测试集的高层表征 target_emotion_feat
    # 冻结全部骨干网络，仅微调 emotion_linear
    for name, p in model.named_parameters():
        if "emotion_linear" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.emotion_linear.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] 启动轻量级分类头快速校准 (Head Refinement, Max Steps: {calib_steps}, LR: {lr})...")
    model.train()
    step = 0
    best_val_acc = 0.0
    best_head_state = None

    train_iter = iter(train_set)
    while step < calib_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_set)
            batch = next(train_iter)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(config.device)

        # 前向计算仅前向骨干以获取 target_emotion_feat
        with torch.no_grad():
            # 统一通过 model.train_one_batch 获取 target_emotion_feat
            _ = model.train_one_batch(batch, 0, train=False)
            # target 特征在 model.current_projected_emotion 或者直接在模型内
        
        # 直接利用前向计算 emotion_loss
        # 打开 emotion_linear 梯度进行快速反向传播
        optimizer.zero_grad()
        # 直接调用模型的前向部分获取 emotion_logits
        # 为精确起见，用 model.train_one_batch 模拟
        bow, kl, mim, ctx, ppl, str_l, str_a, emo_l, emo_a, epcl, dec_emo = model.train_one_batch(batch, 0, train=True)
        # 注意: model.train_one_batch 中在 is_frozen 时 emotion_linear requires_grad=False
        # 我们确保 is_frozen 为 False
        step += 1

        if step % 200 == 0 or step == calib_steps:
            model.eval()
            val_acc_list = []
            with torch.no_grad():
                for v_batch in dev_set:
                    _, _, _, _, _, _, _, _, v_acc, _, _ = model.train_one_batch(v_batch, 0, train=False)
                    val_acc_list.append(v_acc)
            val_acc = np.mean(val_acc_list) * 100
            print(f"  [Step {step}/{calib_steps}] 验证集分类准确率: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_head_state = {k: v.cpu().clone() for k, v in model.emotion_linear.state_dict().items()}
            model.train()

    # 3. 载入最优校准头权重并在测试集上评测
    if best_head_state is not None:
        model.emotion_linear.load_state_dict(best_head_state)
    model.eval()
    all_labels_after, mlp_acc_after = [], []
    with torch.no_grad():
        for batch in tqdm(test_set, desc="Post-Calibration Test Eval"):
            _, _, _, _, _, _, _, _, emo_acc, _, _ = model.train_one_batch(batch, 0, train=False)
            mlp_acc_after.append(emo_acc)
    acc_after = np.mean(mlp_acc_after) * 100

    print(f"\n=======================================================")
    print(f"  分类头校准前后效果对比 (Head Refinement Benchmark)")
    print(f"=======================================================")
    print(f"  校准前测试集准确率: {acc_before:.2f}%")
    print(f"  校准后测试集准确率: {acc_after:.2f}% (净变化: {acc_after - acc_before:+.2f}pp)")
    print(f"=======================================================")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "save/epcl_v8_trial3/CASE_51999_37.1078"
    test_head_calibration(ckpt)
