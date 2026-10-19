# -*- coding: utf-8 -*-
"""B2 前置检查：用 T11 的 ncm_sim 评估原型（Prototypes）的判别几何质量
数据: results/cloud_archive/v9_trial11/dpfc_preds.npz
局限: npz 无原型矩阵本身，故用 ncm_sim（样本×原型余弦）做代理推断
"""
import numpy as np, json

NAMES = ["surprised","excited","annoyed","proud","angry","sad","grateful","lonely","impressed","afraid",
"disgusted","confident","terrified","hopeful","anxious","disappointed","joyful","prepared","guilty","furious",
"nostalgic","jealous","anticipating","embarrassed","content","devastated","sentimental","caring","trusting","ashamed",
"apprehensive","faithful"]

d = np.load(r"E:\github\CASE\results\cloud_archive\v9_trial11\dpfc_preds.npz", allow_pickle=True)
y = d["true_labels"].astype(int)
S = d["ncm_sim"].astype(np.float64)      # [5255, 32] cos(feature, prototype)
N, C = S.shape
print(f"N={N}, C={C}")
print(f"ncm_sim 全局范围: [{S.min():.4f}, {S.max():.4f}]  | 是否已归一化(每行范数≈1): {np.abs(np.linalg.norm(S,axis=1)-1).mean():.4f}")

# ========== 1) Ground-Truth 类间余弦边际 Δsim ==========
rows = np.arange(N)
s_true = S[rows, y]
S_other = S.copy(); S_other[rows, y] = -np.inf
s_best_other = S_other.max(axis=1)
delta = s_true - s_best_other
print("\n===== 1) 真实类边际 Δsim = sim(x,p_y) − max_{j≠y} sim(x,p_j) =====")
print(f"  mean={delta.mean():.4f}  median={np.median(delta):.4f}  std={delta.std():.4f}")
print(f"  P5={np.percentile(delta,5):.4f}  P25={np.percentile(delta,25):.4f}  P75={np.percentile(delta,75):.4f}  P95={np.percentile(delta,95):.4f}")
print(f"  Δsim <= 0 的样本比例: {(delta<=0).mean()*100:.2f}%   (>0 才说明该样本的原型先验是‘正确’的)")
print(f"  Δsim <= 0.05 的比例 : {(delta<=0.05).mean()*100:.2f}%  (边际极薄，先验几乎无指示力)")
print(f"  Δsim <= 0.10 的比例 : {(delta<=0.10).mean()*100:.2f}%")

# ========== 2) 类内 / 类间 相似度分布 ==========
print("\n===== 2) 类内 vs 类间 ncm_sim 分布 =====")
intra = s_true
inter = S_other[rows, S_other.argmax(axis=1)]
print(f"  类内(与真原型)     mean={intra.mean():.4f}  std={intra.std():.4f}")
print(f"  最强类间(最像的错原型) mean={inter.mean():.4f}  std={inter.std():.4f}")
print(f"  两者均值差 = {(intra-inter).mean():.4f}")
print(f"  重叠度：类内 < 最强类间的样本比例 = {(intra < inter).mean()*100:.2f}%")

# ========== 3) NCM 混淆：哪两个原型最易混 ==========
print("\n===== 3) NCM 混淆对 Top 12（按真实类→预测类 计数）=====")
pred = S.argmax(axis=1)
cm = np.zeros((C,C), dtype=int)
for t,p in zip(y,pred):
    cm[t,p]+=1
pairs=[]
for i in range(C):
    for j in range(C):
        if i!=j and cm[i,j]>0:
            pairs.append((cm[i,j], cm[i,i], NAMES[i], NAMES[j]))
pairs.sort(reverse=True)
for cnt, diag, a, b in pairs[:12]:
    rate = cnt/(cm[NAMES.index(a)].sum())*100
    print(f"   {a:<14} -> {b:<14}  n={cnt:>3}  ({rate:.1f}% of {a})   [同类正确={diag}]")

# ========== 4) 每个原型的‘吸引域污染’ ==========
print("\n===== 4) 每个原型的吸引域污染率（被非本类样本错误归入的比例）=====")
contam = []
for j in range(C):
    tot = cm[:,j].sum()
    own = cm[j,j]
    contam.append((1 - own/tot if tot>0 else 0.0, NAMES[j], tot, own))
contam.sort(reverse=True)
print("   污染最重的 10 个原型：")
for r, nm, tot, own in contam[:10]:
    print(f"   {nm:<14} 吸引样本={tot:>3}  本类={own:>3}  污染率={r*100:>5.1f}%")
print("   污染最轻的 5 个原型：")
for r, nm, tot, own in contam[-5:]:
    print(f"   {nm:<14} 吸引样本={tot:>3}  本类={own:>3}  污染率={r*100:>5.1f}%")

# ========== 5) 原型间相似度代理：列向量相关 ==========
print("\n===== 5) 原型相似度代理（ncm_sim 列向量相关系数，非直接 P·P^T）=====")
Z = (S - S.mean(axis=0, keepdims=True)) / (S.std(axis=0, keepdims=True) + 1e-9)
R = (Z.T @ Z) / N
np.fill_diagonal(R, -np.inf)
tri = R[np.triu_indices(C, 1)]
print(f"  非对角列相关：mean={tri.mean():.4f}  max={tri.max():.4f}")
idx = np.dstack(np.unravel_index(np.argsort(R, axis=None)[::-1], R.shape))[0]
seen=set(); top=[]
for i,j in idx:
    if i>=j: continue
    key=(i,j)
    if key in seen: continue
    seen.add(key); top.append((R[i,j], NAMES[i], NAMES[j]))
    if len(top)>=8: break
print("  最相似的 8 对（列相关）：")
for v,a,b in top:
    print(f"   {a:<14} ~ {b:<14}  corr={v:.3f}")

# ========== 6) 结论判定（量化） ==========
d_le0 = (delta<=0).mean()*100
d_le05 = (delta<=0.05).mean()*100
contam_bad = sum(1 for r,_,_,_ in contam if r>0.5)
print("\n===== 6) B2 前置判据（事前约定，见回复正文）=====")
print(f"  [A] Δsim<=0 比例        = {d_le0:.2f}%    (阈值: >50% => 原型先验整体无指示力)")
print(f"  [B] Δsim<=0.05 比例     = {d_le05:.2f}%   (阈值: >50% => 边际过薄)")
print(f"  [C] 污染率>50% 的原型数 = {contam_bad}/32  (阈值: >16 => 过半原型不可分)")
print(f"  [D] NCM 准确率          = {(pred==y).mean()*100:.2f}%")

out = dict(delta_mean=float(delta.mean()), delta_median=float(np.median(delta)),
           frac_le0=float(d_le0), frac_le05=float(d_le05),
           intra_mean=float(intra.mean()), inter_mean=float(inter.mean()),
           ncm_acc=float((pred==y).mean()*100),
           contam_gt50=int(contam_bad))
print("\n[JSON] " + json.dumps(out, ensure_ascii=False))
