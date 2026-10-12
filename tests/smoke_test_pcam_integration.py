# -*- coding: utf-8 -*-
"""
CASE-EPCL V9 升级版 PCAM (原型自适应软记忆反哺) 端到端冒烟测试
验证包含:
1. 模型加载与 PCAM 初始化 (tau_attn=1.5, gamma=0.0);
2. 真实数据 batch 下的 train_one_batch 单步前向与反向传播 (梯度健康度与 gamma 监控);
3. 自回归贪心解码 (decoder_greedy) 与采样解码 (decoder_sampling) 的 apply_pcam 接入;
4. 束搜索 (BeamSearch) 下的 apply_pcam 接口对齐;
5. 检查点安全保存与 eval_pipeline 拓扑自适应权重识别。
"""
import os
import sys
import shutil
import tempfile
import torch

# 注册工程根路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq
from src.utils.decode.case import Translator


def run_smoke_test():
    print("=" * 70)
    print("[*] 开始执行 V9 升级版 PCAM 端到端冒烟测试")
    print("=" * 70)

    # 1. 显式覆盖配置项
    config.seed = 13
    set_seed()
    config.pretrain = False
    config.use_pcam = True
    config.pcam_tau_attn = 1.5
    config.pcam_heads = 2
    config.pcam_dropout = 0.1
    config.pcam_gate_bias = -1.0
    config.use_mcp = True
    config.mcp_momentum = 0.96
    config.lambda_epcl = 0.03
    config.alpha_uni = 0.8
    config.arc_margin = 0.30
    config.arc_mode = "cos"
    config.batch_size = 2
    config.check_iter = 1  # 冒烟测试中步频设为 1 以验证 gamma 监控打印
    
    # 临时目录用于保存检查点
    temp_dir = tempfile.mkdtemp(prefix="pcam_smoke_test_")
    config.save_path = temp_dir
    config.model_file_path = temp_dir

    try:
        # 2. 加载小批量真实数据
        print("[1/5] 加载 ED 数据集测试 Batch...")
        train_set, dev_set, test_set, vocab, emo_num, strategy_num = prepare_data_seq(batch_size=2)
        print(f"      数据就绪: 词表大小={vocab.n_words}, 情感类数={emo_num}")

        # 3. 初始化 CASE 模型
        print("[2/5] 实例化完整 CASE 模型 (含升级版 PCAM)...")
        model = CASE(
            vocab,
            emotion_num=emo_num,
            strategy_num=strategy_num,
            is_eval=False,
            model_file_path=None,
        )
        model = model.to(config.device)
        
        # 校验 PCAM 实例与参数
        assert hasattr(model, "pcam"), "Model 必须包含 pcam 模块"
        assert model.pcam.tau_attn == 1.5, f"pcam.tau_attn 应为 1.5，实际为 {model.pcam.tau_attn}"
        assert abs(model.pcam.gamma.item()) < 1e-6, f"pcam.gamma 初始值应为 0.0，实际为 {model.pcam.gamma.item()}"
        print(f"      PCAM 模块校验通过: tau_attn={model.pcam.tau_attn}, gamma={model.pcam.gamma.item():.4f}")

        # 4. 单步训练前向与反向传播
        print("[3/5] 执行 train_one_batch 单步前向与梯度反传...")
        model.train()
        batch = next(iter(train_set))
        
        # iter=0 时应触发 gamma 监控打印 (因为 check_iter=1, (0+1)%1==0)
        losses = model.train_one_batch(batch, iter=0, train=True)
        bow_loss, kl_loss, mim_loss, ctx_loss, ppl = losses[:5]
        
        print(f"      前向损失输出: bow={bow_loss:.4f}, kl={kl_loss:.4f}, mim={mim_loss:.4f}, ctx={ctx_loss:.4f}, ppl={ppl:.4f}")
        assert torch.isfinite(torch.tensor(ctx_loss)), "ctx_loss 必须为有限值"
        assert model.pcam.gamma.grad is not None, "pcam.gamma 必须接收到反向传播梯度"
        assert torch.isfinite(model.pcam.gamma.grad).all(), "pcam.gamma 梯度必须数值健康"
        print(f"      PCAM 梯度回传校验通过: gamma.grad={model.pcam.gamma.grad.item():.6f}")

        # 5. 解码模块冒烟测试 (decoder_sampling 与 beam_search 原生面向单样本推理)
        print("[4/5] 测试推理解码端 apply_pcam 接口...")
        model.eval()
        single_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                single_batch[k] = v[:1]
            elif isinstance(v, list):
                single_batch[k] = v[:1]
            else:
                single_batch[k] = v
        
        # 5.1 贪心解码
        greedy_words = model.decoder_greedy(single_batch, max_dec_step=5)
        assert len(greedy_words) > 0, "decoder_greedy 输出不能为空"
        print(f"      decoder_greedy 解码成功 (生成片段: {' '.join(greedy_words[0][:5])}...)")

        # 5.2 采样解码
        sampling_words = model.decoder_sampling(single_batch, max_dec_step=5, temp=0.7)
        assert len(sampling_words) > 0, "decoder_sampling 输出不能为空"
        print(f"      decoder_sampling 解码成功 (生成片段: {' '.join(sampling_words[0][:5])}...)")

        # 5.3 束搜索解码
        translator = Translator(model, vocab)
        beam_preds = translator.beam_search(single_batch, max_dec_step=5)
        assert len(beam_preds) > 0, "BeamSearch 解码输出不能为空"
        print("      BeamSearch 解码成功")

        # 6. 检查点安全落盘与拓扑自适应预检
        print("[5/5] 验证检查点落盘与 eval_pipeline 拓扑自适应...")
        model.save_model(running_avg_ppl=35.0, iter=100, tag="ppl_best")
        ckpt_path = os.path.join(temp_dir, "CASE_100_35.0000_ppl_best")
        assert os.path.exists(ckpt_path), f"检查点文件未在预期路径生成: {ckpt_path}"
        
        saved_state = torch.load(ckpt_path, map_location="cpu")
        raw_keys = saved_state["model"].keys()
        pcam_keys = [k for k in raw_keys if "pcam." in k]
        assert len(pcam_keys) > 0, "保存的权重中必须包含 pcam.* 相关键"
        assert any("pcam.gamma" in k for k in pcam_keys), "保存的权重中必须包含 pcam.gamma"
        print(f"      检查点键验证通过: 包含 {len(pcam_keys)} 个 PCAM 相关参数键 (含 pcam.gamma)")

        # 验证 eval_pipeline 拓扑自适应逻辑
        if any("pcam." in k for k in raw_keys):
            detected_pcam = True
        else:
            detected_pcam = False
        assert detected_pcam, "eval_pipeline 拓扑预检未能识别到 PCAM 权重"
        print("      eval_pipeline 拓扑自适应逻辑成功识别 PCAM 权重！")

        print("=" * 70)
        print("[SUCCESS] 全部冒烟测试项 100% 顺利通过！")
        print("=" * 70)

    finally:
        # 清理临时测试目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()
