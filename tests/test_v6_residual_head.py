# -*- coding: utf-8 -*-
"""
CASE-EPCL V6 Trial 1 离线单元测试脚本:
验证 ResidualEmotionHead 双层非线性残差分类头的前向传播、残差梯度连通性、
参数冻结以及与 BCF 黄金权重回滚机制的完全兼容性。
"""
import sys
import os
from copy import deepcopy
import torch
import torch.nn as nn

# 注入项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import config
from src.models.CASE.model import CASE, ResidualEmotionHead

# 强制单元测试使用 CPU 运行，避免机器 GPU 编号不一致
config.cuda = False
config.device = torch.device("cpu")


def test_residual_head_standalone():
    """测试 1: 独立验证 ResidualEmotionHead 模块形状与梯度连通性"""
    print("\n[测试 1] 验证 ResidualEmotionHead 模块独立前向与反向传播...")
    bsz, d_in, d_hid, num_classes = 8, 300, 300, 32
    head = ResidualEmotionHead(input_dim=d_in, hidden_dim=d_hid, num_classes=num_classes, dropout=0.1)
    head.train()
    
    x = torch.randn(bsz, d_in, requires_grad=True)
    out = head(x)
    assert out.shape == (bsz, num_classes), f"输出维度异常: 期望 {(bsz, num_classes)}, 实际 {out.shape}"
    
    # 计算损失并反向传播
    target = torch.randint(0, num_classes, (bsz,))
    loss = nn.CrossEntropyLoss()(out, target)
    loss.backward()
    
    # 验证主干分支与残差捷径分支的梯度均正常回传
    assert head.fc1.weight.grad is not None and torch.norm(head.fc1.weight.grad) > 0, "fc1 梯度丢失！"
    assert head.fc2.weight.grad is not None and torch.norm(head.fc2.weight.grad) > 0, "fc2 梯度丢失！"
    assert head.res_proj.weight.grad is not None and torch.norm(head.res_proj.weight.grad) > 0, "res_proj 残差捷径梯度丢失！"
    assert x.grad is not None and torch.norm(x.grad) > 0, "输入 x 梯度丢失！"
    print("  -> ResidualEmotionHead 独立形状与梯度连通性校验 100% 通过！")


def test_residual_head_freeze_and_rollback():
    """测试 2: 验证多层参数的梯度冻结与 BCF 黄金权重回滚"""
    print("\n[测试 2] 验证 ResidualEmotionHead 多层参数冻结与 BCF 状态回滚...")
    bsz, d_in, d_hid, num_classes = 4, 300, 300, 32
    head = ResidualEmotionHead(input_dim=d_in, hidden_dim=d_hid, num_classes=num_classes, dropout=0.1)
    
    # 记录初始状态作为黄金状态
    golden_state = deepcopy(head.state_dict())
    
    # 模拟训练期间的权重漂移
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
            
    # 验证权重已发生漂移
    drifted_weight_norm = torch.norm(head.fc1.weight - golden_state["fc1.weight"]).item()
    assert drifted_weight_norm > 1e-3, "模拟权重漂移失败！"
    
    # 模拟 BCF 回滚
    head.load_state_dict(golden_state)
    rollback_weight_norm = torch.norm(head.fc1.weight - golden_state["fc1.weight"]).item()
    assert rollback_weight_norm == 0.0, "BCF 权重回滚失败，存在残余差异！"
    
    # 模拟冻结分类头
    for p in head.parameters():
        p.requires_grad = False
        
    for name, p in head.named_parameters():
        assert not p.requires_grad, f"参数 {name} 未被成功冻结为 requires_grad=False！"
        
    print("  -> ResidualEmotionHead 多层参数冻结与 BCF 状态回滚 100% 通过！")


def test_case_model_integration():
    """测试 3: 验证 CASE 主体模型对 ResidualEmotionHead 的集成与向后兼容性"""
    print("\n[测试 3] 验证 CASE 完整模型集成与向后兼容切换...")
    
    class DummyVocab:
        def __init__(self):
            self.n_words = 100
            self.word2index = {'<PAD>': 1}
            self.index2word = {1: '<PAD>'}
            
    dummy_vocab = DummyVocab()
    config.pretrain_emb = False
    
    # 1. 验证向后兼容默认行为 (linear)
    config.emotion_head_type = "linear"
    model_linear = CASE(dummy_vocab, emotion_num=32, strategy_num=8)
    assert isinstance(model_linear.emotion_linear, nn.Linear), "向后兼容模式下未能正确装载 nn.Linear！"
    print("  -> 向后兼容模式 (linear) 校验通过！")
    
    # 2. 验证 V6 全新模式 (residual_mlp)
    config.emotion_head_type = "residual_mlp"
    config.mlp_hidden_dim = 300
    config.mlp_dropout = 0.1
    model_v6 = CASE(dummy_vocab, emotion_num=32, strategy_num=8)
    assert isinstance(model_v6.emotion_linear, ResidualEmotionHead), "V6 模式下未能正确装载 ResidualEmotionHead！"
    
    # 3. 验证通过 model_v6.freeze_emo_head 进行 BCF 黄金回滚与冻结
    golden_head_state = deepcopy(model_v6.emotion_linear.state_dict())
    
    # 扰动
    with torch.no_grad():
        for p in model_v6.emotion_linear.parameters():
            p.add_(torch.ones_like(p))
            
    # 执行冻结与回滚
    model_v6.freeze_emo_head(step=26000, best_state=golden_head_state, best_step=20000)
    assert model_v6.is_frozen is True, "模型冻结标志位未置为 True！"
    assert model_v6.actual_freeze_step == 26000, "模型实际冻结步数未正确更新！"
    
    # 检查权重恢复与梯度锁定
    recovered_norm = torch.norm(model_v6.emotion_linear.fc1.weight - golden_head_state["fc1.weight"]).item()
    assert recovered_norm == 0.0, "通过 CASE 模型回滚分类头权重失败！"
    
    for name, p in model_v6.emotion_linear.named_parameters():
        assert not p.requires_grad, f"模型分类头子参数 {name} 未被冻结！"
        
    print("  -> CASE 模型集成、BCF 黄金回滚与自适应冻结状态机 100% 通过！")


if __name__ == "__main__":
    print("=" * 65)
    print("开始执行 CASE-EPCL V6 Trial 1 (ResidualEmotionHead) 单元测试")
    print("=" * 65)
    
    test_residual_head_standalone()
    test_residual_head_freeze_and_rollback()
    test_case_model_integration()
    
    print("\n" + "=" * 65)
    print("恭喜！所有 V6 Trial 1 单元测试 100% 通过！系统已具备启动实测条件！")
    print("=" * 65)
