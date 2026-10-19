import re

def extract_valid(log_file):
    records = []
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('valid\t'):
                parts = line.strip().split('\t')
                records.append({
                    'step': (len(records) + 1) * 500 - 1,
                    'ppl': float(parts[5]),
                    'emo_loss': float(parts[8]),
                    'emo_acc': float(parts[9])
                })
    return records

t11_a = extract_valid('results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam.log')
t11 = extract_valid('results/cloud_archive/v9_trial11/v9_trial11_seed13.log')

print(f"Length: T11-A={len(t11_a)}, T11={len(t11)}")
diffs_ppl = []
diffs_acc = []
print(f"{'Step':<6} | {'T11 PPL':<8} {'T11A PPL':<9} {'Delta PPL':<10} | {'T11 Acc':<8} {'T11A Acc':<9} {'Delta Acc':<10}")
for i in range(min(len(t11), len(t11_a))):
    s11 = t11[i]
    s11a = t11_a[i]
    d_ppl = s11a['ppl'] - s11['ppl']
    d_acc = (s11a['emo_acc'] - s11['emo_acc']) * 100
    diffs_ppl.append(d_ppl)
    diffs_acc.append(d_acc)
    print(f"{s11['step']:<6} | {s11['ppl']:<8.2f} {s11a['ppl']:<9.2f} {d_ppl:<+10.2f} | {s11['emo_acc']*100:<7.2f}% {s11a['emo_acc']*100:<8.2f}% {d_acc:<+10.2f}pp")

print("PPL worse count (d_ppl > 0):", sum(1 for d in diffs_ppl if d > 0), "/", len(diffs_ppl))
print("Acc worse count (d_acc < 0):", sum(1 for d in diffs_acc if d < 0), "/", len(diffs_acc))
