"""Trial12检查点模式约束及隔离训练状态的前缀诊断。"""
import contextlib
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PREFIX_NORM_MODE = "shared_gamma_std_no_beta_v1"
FROZEN_INIT_SHA256 = {
    "proto": "f061c61c9499c202e82586018cf6755364683c434ca013eafe6ec8b20d414169",
    "random": "c079bcb310653061f3c704140eda6d367dffdaf3d309178066b571a355c0de94",
}


def checkpoint_prefix_norm(checkpoint):
    """旧权重默认关闭；新权重的模式元数据与各层标记必须一致。"""
    state = checkpoint.get("model", checkpoint)
    prefixes = {k[:-len("prefix_embed")] for k in state if k.endswith("prefix_embed")}
    markers = {k[:-len("prefix_norm_version")] for k in state if k.endswith("prefix_norm_version")}
    mode = checkpoint.get("prefix_norm_mode")
    if mode not in (None, "none", PREFIX_NORM_MODE):
        raise RuntimeError("Unknown checkpoint prefix_norm_mode: " + str(mode))
    enabled = mode == PREFIX_NORM_MODE
    if not enabled:
        if markers:
            raise RuntimeError("Prefix norm marker without matching metadata")
        return False
    if not prefixes or markers != prefixes:
        raise RuntimeError("Prefix norm metadata/marker topology mismatch")
    for prefix in prefixes:
        value = state[prefix + "prefix_norm_version"]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                or value.ndim != 0 or value.item() != 1 or prefix + "prefix_gate" not in state):
            raise RuntimeError("Invalid prefix norm marker or missing gate: " + prefix)
    if checkpoint.get("prefix_attention_mode") != "separate_softmax_residual":
        raise RuntimeError("Normalized prefix requires declared residual attention mode")
    return True


def capture_rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def assert_rng_equal(before, after):
    assert before[0] == after[0], "Python RNG changed"
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:], "NumPy RNG changed"
    assert torch.equal(before[2], after[2]), "CPU torch RNG changed"
    assert len(before[3]) == len(after[3]), "CUDA RNG device set changed"
    assert all(torch.equal(a, b) for a, b in zip(before[3], after[3])), "CUDA RNG changed"


@contextlib.contextmanager
def isolated_rng():
    """保存并恢复四路随机状态，观测异常退出时也执行恢复与断言。"""
    state = capture_rng_state()
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3]:
            torch.cuda.set_rng_state_all(state[3])
        assert_rng_equal(state, capture_rng_state())


@contextlib.contextmanager
def isolated_model_observation(model):
    """旧评测会更新词频权重并清理梯度；新增观测同时隔离这些副作用。"""
    modes = [(module, module.training) for module in model.modules()]
    params = [(p, p._version, p.grad, p.grad.detach().clone() if p.grad is not None else None)
              for p in model.parameters()]
    buffers = [(m, name, b, b.detach().clone()) for m in model.modules()
               for name, b in m._buffers.items() if b is not None]
    attrs = ("word_freq", "current_pred_emotion", "current_projected_emotion", "current_emo_vocab_bias")
    present = {k: hasattr(model, k) for k in attrs}
    saved = {k: copy.deepcopy(getattr(model, k)) for k in attrs if present[k]}
    try:
        with isolated_rng(), torch.no_grad():
            model.eval()
            yield
    finally:
        bad_buffers = []
        criterion = getattr(model, "criterion", None)
        for m, name, ref, value in buffers:
            current = m._buffers.get(name)
            # 旧评测路径会更新criterion.weight；允许发生后恢复原值。
            if not (m is criterion and name == "weight"):
                if current is None or not torch.equal(current, value):
                    bad_buffers.append(type(m).__name__ + "." + name)
            ref.copy_(value)
            m._buffers[name] = ref
        for k in attrs:
            if present[k]:
                setattr(model, k, saved[k])
            elif hasattr(model, k):
                delattr(model, k)
        for p, version, ref, value in params:
            if ref is not None:
                ref.copy_(value)
            p.grad = ref
        for module, training in modes:
            module.training = training
        if bad_buffers:
            raise RuntimeError("Diagnostic unexpectedly changed model buffers: " + str(bad_buffers))
        if any(p._version != version for p, version, _, _ in params):
            raise RuntimeError("Diagnostic unexpectedly changed model parameters")


def _entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)


def emotion_selection_summary(example_attention, labels, minimum_count=5):
    """样本先等权平均，再按合格类别的实际样本占比计算熵差。"""
    labels = np.asarray(labels)
    groups = []
    for label in sorted(set(labels.tolist())):
        keep = torch.from_numpy(labels == label)
        groups.append({"label": int(label), "n": int(keep.sum()),
                       "mean_attention": example_attention[keep].mean(0)})
    eligible = [g for g in groups if g["n"] >= minimum_count]
    covered = sum(g["n"] for g in eligible)
    difference = None
    if len(eligible) >= 2:
        weights = torch.tensor([g["n"] / covered for g in eligible], dtype=torch.float64)
        means = torch.stack([g["mean_attention"] for g in eligible]).double()
        overall = (weights[:, None, None] * means).sum(0)
        difference = (_entropy(overall) - (weights[:, None] * _entropy(means)).sum(0)).tolist()
    return {"aggregation": "sample macro; empirical class frequency weights within eligible subset",
            "minimum_class_n": minimum_count, "eligible_classes": len(eligible),
            "covered_n": covered, "total_n": len(labels), "entropy_gap_by_head": difference,
            "classes": [{**g, "mean_attention": g["mean_attention"].tolist()} for g in groups],
            "interpretation": "descriptive label association, no semantic/causal claim or selection threshold"}


def _summary(values):
    x = torch.cat(values).double()
    if not x.numel() or not torch.isfinite(x).all():
        raise RuntimeError("Empty or non-finite prefix diagnostic")
    return {"n_observations": x.numel(), "mean": x.mean().item(),
            "median": torch.quantile(x, .5).item(), "p95": torch.quantile(x, .95).item()}


def observe_prefix(model, valid_loader, step, output_path=None, sample_n=256, batch_size=4):
    """只读固定Valid切片；保留旧全量Valid流程的既有行为。"""
    if not getattr(model, "use_proto_prefix_gate", False):
        raise ValueError("Prefix diagnostic requires gated prefix")
    if sample_n > len(valid_loader.dataset):
        raise ValueError("Insufficient validation examples for requested diagnostic scope")
    layers = {name: layer for name, layer in model.named_modules()
              if getattr(layer, "use_proto_prefix_gate", False) and hasattr(layer, "multi_head_attention_dec")}
    if not layers:
        raise RuntimeError("No gated DecoderLayer found")
    data, current, previous = {}, {}, {}
    for name, layer in layers.items():
        data[name] = {"weights": [], "examples": [], "labels": [], "ratio": [],
                      "text_norm": [], "prefix_norm": [], "output_ratio": [], "small_denominator_n": 0}
    seen = 0
    with isolated_model_observation(model):
        try:
            for name, layer in layers.items():
                attn = layer.multi_head_attention_dec
                previous[name] = getattr(attn, "_prefix_observer", None)

                def capture(weights, text, prefix, gate, name=name, attn=attn):
                    item = data[name]
                    keep = current["keep"]
                    weights = weights.detach()
                    ok = keep[:, None, :].expand(text.shape[:-1])
                    norm = text.norm(dim=-1)
                    residual = gate.view(1, -1, 1, 1) * prefix
                    item["ratio"].append((residual.norm(dim=-1) / norm.clamp_min(1e-12))[ok].cpu())
                    item["text_norm"].append(norm[ok].cpu())
                    item["prefix_norm"].append(prefix.norm(dim=-1)[ok].cpu())
                    item["small_denominator_n"] += int((norm[ok] < 1e-12).sum())
                    # [B,H,T,P]转为[非padding query,H,P]。
                    item["weights"].append(weights.permute(0, 2, 1, 3)[keep].cpu())
                    lengths = keep.sum(-1)
                    if (lengths == 0).any():
                        raise RuntimeError("No nonpadding target in diagnostic example")
                    mean = (weights * keep[:, None, :, None]).sum(2) / lengths[:, None, None]
                    item["examples"].append(mean.cpu())
                    item["labels"].extend(current["labels"])
                    projected_text = attn.output_linear(attn._merge_heads(text))
                    projected_residual = attn.output_linear(attn._merge_heads(residual))
                    item["output_ratio"].append((projected_residual.norm(dim=-1) /
                        projected_text.norm(dim=-1).clamp_min(1e-12))[keep].cpu())

                attn._prefix_observer = capture
            loader = DataLoader(Subset(valid_loader.dataset, range(sample_n)), batch_size=batch_size,
                                shuffle=False, num_workers=0, collate_fn=valid_loader.collate_fn,
                                generator=torch.Generator().manual_seed(13))
            # 在隔离区内导入，避免config导入影响调用方随机状态。
            from src.utils.config import config
            for batch in loader:
                current["keep"] = batch["target_batch"].ne(config.PAD_idx)
                current["labels"] = batch["program_label"].detach().cpu().tolist()
                model.train_one_batch(batch, step, train=False)
                seen += len(current["labels"])
        finally:
            for name, callback in previous.items():
                layers[name].multi_head_attention_dec._prefix_observer = callback
    assert seen == sample_n
    result = {"step": step, "scope": {"split": "Valid sequential first-N", "evaluated_n": seen,
              "batch_size": batch_size, "teacher_forcing": True, "training": False},
              "rng_restored_exact": True, "layers": {}}
    for name, item in data.items():
        weights = torch.cat(item["weights"]).double()
        ent = _entropy(weights)
        layer = layers[name]
        result["layers"][name] = {
            "norm_mode": PREFIX_NORM_MODE if layer.use_proto_prefix_norm else "none",
            "tanh_gate": layer.prefix_gate.detach().tanh().cpu().tolist(),
            "nonpadding_queries": weights.size(0),
            "entropy": _summary([ent.reshape(-1)]),
            "entropy_by_head": [_summary([ent[:, h]]) for h in range(ent.size(1))],
            "max_attention": _summary([weights.max(-1).values.reshape(-1)]),
            "query_js_by_head": (_entropy(weights.mean(0)) - ent.mean(0)).tolist(),
            "residual_to_text_ratio": _summary(item["ratio"]),
            "text_context_norm": _summary(item["text_norm"]),
            "prefix_context_norm": _summary(item["prefix_norm"]),
            "projected_residual_to_text_ratio": _summary(item["output_ratio"]),
            "small_denominator_n": item["small_denominator_n"],
            "emotion_selection": emotion_selection_summary(torch.cat(item["examples"]), item["labels"]),
        }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    concise = {name: {k: record[k] for k in ("norm_mode", "tanh_gate", "entropy",
                "query_js_by_head", "residual_to_text_ratio")} for name, record in result["layers"].items()}
    print("[Prefix Diagnostic] " + json.dumps({"step": step, "scope": result["scope"],
          "rng_restored_exact": True, "layers": concise}, ensure_ascii=False))
    return result
