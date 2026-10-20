# -*- coding: utf-8 -*-
"""Read-only Trial 12 dry-run artifact audit; never treats skipped Test as measured.
Run from repo root: python src/scripts/audit_trial12_dry_run.py --output results/v9/trial12_handoff_audit/dry_run.json
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def log_evidence(path, needles):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    return [{"line": i, "text": line.strip()} for i, line in enumerate(text.split("\n"), 1)
            if any(needle in line for needle in needles)]


def train_curve(directory, metric):
    files = list((directory / metric / "loss_train").glob("events.*"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one fresh loss_train event file: %s" % files)
    events = EventAccumulator(str(files[0]), size_guidance={"scalars": 0}).Reload()
    scalar_tags = events.Tags()["scalars"]
    if len(scalar_tags) != 1:
        raise RuntimeError("Ambiguous scalar tags: %s" % scalar_tags)
    rows = [{"step": event.step, "value": event.value} for event in events.Scalars(scalar_tags[0])]
    if [row["step"] for row in rows] != list(range(10)):
        raise RuntimeError("Expected exactly steps 0..9, got: %s" % rows)
    if not all(math.isfinite(row["value"]) for row in rows):
        raise RuntimeError("Non-finite training scalar")
    return {"path": str(files[0].relative_to(ROOT)), "tag": scalar_tags[0], "values": rows}


def checkpoint_info(directory, initial):
    candidates = list(directory.glob("CASE_*_composite_best"))
    if len(candidates) != 1:
        raise RuntimeError("Expected one composite checkpoint: %s" % candidates)
    path = candidates[0]
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    state = torch.load(str(path), **kwargs)
    sd = state.get("model", state)
    keys = [k for k in sd if k.endswith("prefix_embed")]
    if len(keys) != 1:
        raise RuntimeError("Expected one prefix tensor: %s" % keys)
    prefix = sd[keys[0]].detach().cpu().numpy().copy()
    result = {"path": str(path.relative_to(ROOT)), "step": int(path.name.split("_")[1]),
              "md5": digest(path, "md5"), "sha256": digest(path), "nlink": path.stat().st_nlink,
              "prefix_key": keys[0], "prefix_shape": list(prefix.shape),
              "prefix_l2": np.linalg.norm(prefix, axis=1).tolist(),
              "max_abs_delta_from_initial_after_training": float(np.max(np.abs(prefix - initial)))}
    del state, sd
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on_dir", default="save/v9_trial12_on")
    parser.add_argument("--random_dir", default="save/v9_trial12_rand")
    parser.add_argument("--off_log", default="logs/v9_trial12_off.log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    on, random = ROOT / args.on_dir, ROOT / args.random_dir
    npz = on / "proto_prefix_init.npz"
    with np.load(str(npz)) as data:
        initial = data["init"].copy()
    if initial.shape != (4, 300) or initial.dtype != np.float32 or not np.isfinite(initial).all():
        raise RuntimeError("Invalid initialization NPZ")
    generator = torch.Generator().manual_seed(13)
    e = torch.from_numpy(initial)
    r = torch.randn(e.shape, generator=generator) * e.std() + e.mean()
    r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8) * e.norm(dim=1, keepdim=True)
    report = {"scope": "code-path dry run only; no formal Test or generation-quality conclusion",
              "formal_protocol": {"test": {"n": 5255, "batch_size": 1, "aggregation": "exp(mean per-sample token CE), macro"},
                                  "valid": {"n": 5734, "batch_size": 32, "shuffle": False, "aggregation": "exp(mean per-batch token CE), macro"},
                                  "generation": {"n": 500, "batch_size": 1}},
              "dry_protocol": {"train_micro_steps": 10, "train_batch_size": 4, "accum_steps": 8,
                               "optimizer_updates": 1, "valid_n": 5734, "valid_batch_size": 4,
                               "valid_evaluation_points": 2, "test_evaluated_n": 0, "generation_n": 1},
              "initialization": {"path": str(npz.relative_to(ROOT)), "sha256": digest(npz), "shape": list(initial.shape),
                                 "prior_vs_random_initial_max_abs_delta": float((e-r).abs().max()),
                                 "row_l2_alignment_max_abs_delta": float((e.norm(dim=1)-r.norm(dim=1)).abs().max())},
              "runs": {}, "checks": {}}
    needles = ["[V9 T12]", "[Gradient", "[Weight Audit]", "自适应拓扑", "--skip_test", "--skip_manifold", "valid\t", "# PARAMETERS"]
    for name, directory, init in (("on", on, initial), ("random", random, r.numpy())):
        evidence = log_evidence(ROOT / "logs" / (directory.name + ".log"), needles)
        report["runs"][name] = {"checkpoint": checkpoint_info(directory, init),
                                  "ctx_loss": train_curve(directory, "ctx_loss"),
                                  "bow_loss": train_curve(directory, "bow_loss"),
                                  "log": "logs/" + directory.name + ".log", "evidence": evidence}
    on_run, rand_run = report["runs"]["on"], report["runs"]["random"]
    on_text = "\n".join(row["text"] for row in on_run["evidence"])
    rand_text = "\n".join(row["text"] for row in rand_run["evidence"])
    off_evidence = log_evidence(ROOT / args.off_log, ["[V9 T12]", "# PARAMETERS"])
    report["off_evidence"] = {"log": args.off_log, "matches": off_evidence}
    report["checks"] = {
        "checkpoint_md5_different": on_run["checkpoint"]["md5"] != rand_run["checkpoint"]["md5"],
        "ctx_loss_curves_different": on_run["ctx_loss"]["values"] != rand_run["ctx_loss"]["values"],
        "bow_loss_curves_different": on_run["bow_loss"]["values"] != rand_run["bow_loss"]["values"],
        "on_has_prior_and_gram": "prior=" in on_text and "先验原型 Gram 非对角" in on_text,
        "on_has_four_clusters": sum("proto-cluster" in row["text"] for row in on_run["evidence"]) == 4,
        "on_parameter_count": "新增参数=1200" in on_text and "总参数=30,337,457" in on_text,
        "random_parameter_count": "新增参数=1200" in rand_text and "总参数=30,337,457" in rand_text,
        "on_eval_detects_prefix": "检测到原型前缀权重 (P=4)" in on_text,
        "random_eval_detects_prefix": "检测到原型前缀权重 (P=4)" in rand_text,
        "off_no_t12_tag": not any("[V9 T12]" in row["text"] for row in off_evidence),
        "off_parameter_count": any("30336257" in row["text"] for row in off_evidence),
        "initial_l2_aligned": report["initialization"]["row_l2_alignment_max_abs_delta"] < 1e-5}
    report["artifact_checks_passed"] = all(report["checks"].values())
    report["p9_note"] = "Check zero drift after make_model, before optimizer.step (tests/test_trial12_prefix.py). Trained checkpoint drift is expected; do not freeze prefix to force equality."
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "artifact_checks_passed": report["artifact_checks_passed"],
                      "checks": report["checks"]}, indent=2, ensure_ascii=False))
    return 0 if report["artifact_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
