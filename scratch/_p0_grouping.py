# -*- coding: utf-8 -*-
"""Trial 12 · P0 零成本分组诊断（Gate G4）
数据源: results/cloud_archive/v9_trial11/dpfc_preds.npz (N=5255, 32 类)
用途  : 仅作 limitation 诊断，严禁将合并后 Acc 作为成绩申报（铁律 III）
"""
import json, numpy as np

NAMES = ["surprised","excited","annoyed","proud","angry","sad","grateful","lonely","impressed","afraid",
"disgusted","confident","terrified","hopeful","anxious","disappointed","joyful","prepared","guilty","furious",
"nostalgic","jealous","anticipating","embarrassed","content","devastated","sentimental","caring","trusting","ashamed",
"apprehensive","faithful"]
IDX = {n: i for i, n in enumerate(NAMES)}
import os
NPZ = os.path.join("results", "cloud_archive", "v9_trial11", "dpfc_preds.npz")
OUT = os.path.join("results", "v9", "trial12_p0_grouping_analysis.json")

d = np.load(NPZ, allow_pickle=True)
y, PL, PN = d["true_labels"], d["preds_linear"], d["preds_ncm"]
N = int(len(y))
base_l = float((PL == y).mean() * 100); base_n = float((PN == y).mean() * 100)

def make_find(pairs):
    parent = list(range(len(NAMES)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    return find

def acc_with(pairs, preds):
    find = make_find(pairs)
    ok = 0
    for t, p in zip(y, preds):
        if find(int(t)) == find(int(p)): ok += 1
    return ok / N * 100.0

# 零差类精查
zero, nonzero = [], []
for i, name in enumerate(NAMES):
    m = (y == i); n = int(m.sum())
    if n == 0: continue
    la = float((PL[m] == i).mean() * 100); na = float((PN[m] == i).mean() * 100)
    rec = {"class": name, "n": n, "linear_acc": round(la, 2), "ncm_acc": round(na, 2), "diff_pp": round(na - la, 4)}
    (zero if abs(la - na) < 1e-12 else nonzero).append(rec)

SCHEMES = [
    ("P0.1_single_pair_sentimental_nostalgic", [("sentimental","nostalgic")]),
    ("P0.2_two_pairs", [("sentimental","nostalgic"),("angry","furious")]),
    ("P0.3_four_pairs", [("sentimental","nostalgic"),("angry","furious"),
                         ("afraid","terrified"),("apprehensive","anxious")]),
    ("P0.4_ten_pairs", [("sentimental","nostalgic"),("angry","furious"),("afraid","terrified"),
                        ("apprehensive","anxious"),("devastated","sad"),("ashamed","guilty"),
                        ("anticipating","excited"),("impressed","proud"),("confident","prepared"),
                        ("angry","annoyed")]),
]
merge_curve = []
for name, ps in SCHEMES:
    p2 = [(IDX[a], IDX[b]) for a, b in ps]
    al = acc_with(p2, PL); an = acc_with(p2, PN)
    merge_curve.append({"scheme": name, "n_pairs": len(ps), "merged_K": 32 - len(ps),
        "pairs": [list(x) for x in ps],
        "linear_acc": round(al, 4), "linear_delta_pp": round(al - base_l, 4),
        "ncm_acc": round(an, 4), "ncm_delta_pp": round(an - base_n, 4)})

# ★ 随机合并零模型对照（控制"合并本身就是涨分数学操作"的体积效应）
rng = np.random.default_rng(13)
def rand_pairs(k, r):
    perm = r.tolist(); return [(perm[2*i], perm[2*i+1]) for i in range(k)]
null = {}
for k in (1, 2, 4, 6):
    vals_l, vals_n = [], []
    for _ in range(200):
        ps = rand_pairs(k, rng.permutation(len(NAMES)))
        vals_l.append(acc_with(ps, PL)); vals_n.append(acc_with(ps, PN))
    null["k%d" % k] = {
        "k_pairs": k,
        "linear_mean": round(float(np.mean(vals_l)), 4), "linear_std": round(float(np.std(vals_l)), 4),
        "linear_p95": round(float(np.percentile(vals_l, 95)), 4),
        "ncm_mean": round(float(np.mean(vals_n)), 4), "ncm_std": round(float(np.std(vals_n)), 4),
        "ncm_p95": round(float(np.percentile(vals_n, 95)), 4),
        "n_trials": 200, "seed": 13}

# sentimental/nostalgic 细节
mS = (y == IDX["sentimental"]); mN = (y == IDX["nostalgic"])
detail = {
    "sentimental": {"n": int(mS.sum()), "correct_pct": round(float((PL[mS] == IDX["sentimental"]).mean()*100), 2),
                    "confused_as_nostalgic_pct": round(float((PL[mS] == IDX["nostalgic"]).mean()*100), 2)},
    "nostalgic": {"n": int(mN.sum()), "correct_pct": round(float((PL[mN] == IDX["nostalgic"]).mean()*100), 2),
                  "confused_as_sentimental_pct": round(float((PL[mN] == IDX["sentimental"]).mean()*100), 2)}}

out = {
  "meta": {"gate": "G4", "trial": "T12-P0", "source_npz": "results/cloud_archive/v9_trial11/dpfc_preds.npz",
           "n_samples": N, "n_classes": 32, "generated_by": "P0 zero-cost grouping diagnostic",
           "discipline": "仅作 limitation 诊断；严禁将合并后 Acc 作为成绩申报（铁律 III 口径显式化）"},
  "baseline": {"linear_acc": round(base_l, 4), "ncm_acc": round(base_n, 4)},
  "zero_diff_classes": {"count": len(zero), "classes": zero, "nonzero_count": len(nonzero)},
  "merge_curve": merge_curve,
  "randomized_merge_null_model": null,
  "pair_detail": detail,
  "caveat": ("合并 Acc 随 K 单调上升且上界为 100%（全并为一类即 100%），"
             "故单一合并数字不构成表征质量证据；必须与 randomized_merge_null_model 对照解读。"),
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("已落盘:", OUT)
print("基线 linear=%.4f%% ncm=%.4f%%  零差类=%d" % (base_l, base_n, len(zero)))
print("\n-- 实际语义对合并 --")
for r_ in merge_curve:
    print("  %-42s K=%2d  linear=%.2f%% (%+.2f)  ncm=%.2f%% (%+.2f)" % (
        r_["scheme"], r_["merged_K"], r_["linear_acc"], r_["linear_delta_pp"], r_["ncm_acc"], r_["ncm_delta_pp"]))
print("\n-- 随机合并零模型对照（200 次/trial, seed=13）--")
for k, v in null.items():
    print("  随机 %d 对: linear mean=%.2f%% (sd=%.2f, p95=%.2f) | ncm mean=%.2f%% (sd=%.2f, p95=%.2f)" % (
        v["k_pairs"], v["linear_mean"], v["linear_std"], v["linear_p95"],
        v["ncm_mean"], v["ncm_std"], v["ncm_p95"]))
