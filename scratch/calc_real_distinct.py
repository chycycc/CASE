import os
from nltk import word_tokenize

def calc_distinct_n(n, candidates):
    d = {}
    total = 0
    tokenized = [word_tokenize(c) for c in candidates]
    for s in tokenized:
        for i in range(len(s) - n + 1):
            key = tuple(s[i : i + n])
            d[key] = 1
            total += 1
    return (len(d) / (total + 1e-16)) * 100.0

def evaluate_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    greedy_cands = []
    beam_cands = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Greedy:"):
                greedy_cands.append(line[7:].strip())
            elif line.startswith("Beam:"):
                beam_cands.append(line[5:].strip())
                
    n_total = len(greedy_cands)
    if n_total == 0:
        return None
        
    res = {
        "n_total": n_total,
        # Full test set
        "greedy_full_d1": calc_distinct_n(1, greedy_cands),
        "greedy_full_d2": calc_distinct_n(2, greedy_cands),
        "beam_full_d1": calc_distinct_n(1, beam_cands) if beam_cands else 0.0,
        "beam_full_d2": calc_distinct_n(2, beam_cands) if beam_cands else 0.0,
        # N=500 sample
        "greedy_500_d1": calc_distinct_n(1, greedy_cands[:500]),
        "greedy_500_d2": calc_distinct_n(2, greedy_cands[:500]),
        "beam_500_d1": calc_distinct_n(1, beam_cands[:500]) if beam_cands else 0.0,
        "beam_500_d2": calc_distinct_n(2, beam_cands[:500]) if beam_cands else 0.0,
    }
    return res

targets = [
    ("原论文 CASE 报告值", None),
    ("Trial 9", "results/v9/v9_trial9_results.txt"),
    ("Trial 10", "results/cloud_archive/v9_trial10/v9_trial10_results.txt"),
    ("Trial 11", "results/cloud_archive/v9_trial11/v9_trial11_seed13_results.txt"),
    ("Trial 11-A (w/o PCAM)", "results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam_results.txt"),
    ("Trial 11-B (clip=6.0)", "results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6_results.txt"),
]

print(f"{'配置':<25} | {'N':<5} | {'Full Greedy D-1':<15} | {'Full Greedy D-2':<15} | {'Full Beam D-2':<15} | {'500 Greedy D-2':<15} | {'500 Beam D-2':<15}")
print("-" * 125)
print(f"{'原论文 CASE 报告值':<25} | {'5255':<5} | {'0.74%':<15} | {'4.01%':<15} | {'—':<15} | {'—':<15} | {'—':<15}")

for name, path in targets[1:]:
    r = evaluate_file(path)
    if r:
        print(f"{name:<25} | {r['n_total']:<5} | {r['greedy_full_d1']:.2f}%{'':<10} | {r['greedy_full_d2']:.2f}%{'':<10} | {r['beam_full_d2']:.2f}%{'':<10} | {r['greedy_500_d2']:.2f}%{'':<10} | {r['beam_500_d2']:.2f}%{'':<10}")
