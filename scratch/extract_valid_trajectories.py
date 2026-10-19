import re

def extract_valid_metrics(log_file):
    records = []
    step = None
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # 找 step:
            m_step = re.search(r'step:\s*(\d+)', line)
            if m_step:
                step = int(m_step.group(1))
            elif 'Train Epoch' in line:
                m_ep = re.search(r'Epoch\s*(\d+)', line)
            
            if line.startswith('valid\t'):
                parts = line.strip().split('\t')
                if len(parts) >= 10:
                    bow = float(parts[1])
                    kl = float(parts[2])
                    mim = float(parts[3])
                    ctx = float(parts[4])
                    ppl = float(parts[5])
                    emo_loss = float(parts[8])
                    emo_acc = float(parts[9])
                    records.append({
                        'step': (len(records) + 1) * 500 - 1, # 499, 999, 1499...
                        'ppl': ppl,
                        'emo_loss': emo_loss,
                        'emo_acc': emo_acc
                    })
    return records

t10_recs = extract_valid_metrics('results/cloud_archive/v9_trial10/v9_trial10.log')
t11_recs = extract_valid_metrics('results/cloud_archive/v9_trial11/v9_trial11_seed13.log')
t11b_recs = extract_valid_metrics('results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6.log')

print(f"{'Step':<6} | {'T10 PPL':<8} {'T10 Acc':<8} | {'T11 PPL':<8} {'T11 Acc':<8} | {'T11-B PPL':<9} {'T11-B Acc':<9}")
print("-" * 65)

for i in range(max(len(t10_recs), len(t11_recs), len(t11b_recs))):
    s10 = t10_recs[i] if i < len(t10_recs) else None
    s11 = t11_recs[i] if i < len(t11_recs) else None
    s11b = t11b_recs[i] if i < len(t11b_recs) else None
    
    step = (i + 1) * 500 - 1
    p10 = f"{s10['ppl']:.2f}" if s10 else "-"
    a10 = f"{s10['emo_acc']*100:.2f}%" if s10 else "-"
    p11 = f"{s11['ppl']:.2f}" if s11 else "-"
    a11 = f"{s11['emo_acc']*100:.2f}%" if s11 else "-"
    p11b = f"{s11b['ppl']:.2f}" if s11b else "-"
    a11b = f"{s11b['emo_acc']*100:.2f}%" if s11b else "-"
    
    print(f"{step:<6} | {p10:<8} {a10:<8} | {p11:<8} {a11:<8} | {p11b:<9} {a11b:<9}")
