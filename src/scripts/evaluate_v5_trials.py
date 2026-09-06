import os
import sys
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
    return len(d) / (total + 1e-16) * 100

def parse_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split('\t')
    row = lines[1].strip().split('\t')
    metric_map = dict(zip(header, row))
    
    beams = []
    greedies = []
    refs = []
    
    for line in lines[2:]:
        if line.startswith('Beam:'):
            beams.append(line.replace('Beam:', '').strip())
        elif line.startswith('Greedy:'):
            greedies.append(line.replace('Greedy:', '').strip())
        elif line.startswith('Ref:'):
            refs.append(line.replace('Ref:', '').strip())
            
    # 计算 Distinct
    g_d1 = calc_distinct_n(1, greedies)
    g_d2 = calc_distinct_n(2, greedies)
    b_d1 = calc_distinct_n(1, beams)
    b_d2 = calc_distinct_n(2, beams)
    # 计算唯一响应率 (Unique Response Rate)
    unique_rate = len(set(greedies)) / (len(greedies) + 1e-16) * 100
    
    ppl = float(metric_map.get('PPL', 0))
    emo_acc = float(metric_map.get('EMO_acc', 0)) * 100
    bow_loss = float(metric_map.get('BOW_Loss', 0))
    emo_loss = float(metric_map.get('EMO_loss', 0))
    
    return {
        'PPL': ppl,
        'Emo_Acc': emo_acc,
        'BOW_Loss': bow_loss,
        'EMO_Loss': emo_loss,
        'G_Dist1': g_d1,
        'G_Dist2': g_d2,
        'B_Dist1': b_d1,
        'B_Dist2': b_d2,
        'Unique': unique_rate,
        'num_samples': len(refs)
    }

if __name__ == '__main__':
    trials = [
        ('V4 Trial 8 (linear, 28k fix)', 'save/epcl_v4_mop_dr_trial8/results.txt'),
        ('V5 Trial 1 (cosine, 28k fix)', 'save/epcl_v5_trial1/results.txt'),
        ('V5 Trial 2b (alpha=0.08)', 'save/epcl_v5_trial2b/results.txt'),
        ('V5 Trial 2c (alpha=0.12)', 'save/epcl_v5_trial2c/results.txt'),
        ('V5 Trial 3a (div=1.8)', 'save/epcl_v5_trial3a/results.txt'),
        ('V5 Trial 3b (div=2.2)', 'save/epcl_v5_trial3b/results.txt'),
        ('V5 Trial 3c (div=2.5)', 'save/epcl_v5_trial3c/results.txt'),
        ('V5 Trial 4a (Adaptive Freeze: 26k)', 'save/epcl_v5_trial4a/results.txt'),
        ('V5 Trial 4b (ACF + BCF: 20k)', 'save/epcl_v5_trial4b/results.txt'),
    ]

    header_str = f"| {'Model / Trial':<35} | {'PPL':<7} | {'Emo Acc%':<9} | {'G-Dist1':<8} | {'G-Dist2':<8} | {'B-Dist1':<8} | {'B-Dist2':<8} | {'Unique%':<8} |"
    sep_str = '|' + '-'*37 + '|' + '-'*9 + '|' + '-'*11 + '|' + '-'*10 + '|' + '-'*10 + '|' + '-'*10 + '|' + '-'*10 + '|' + '-'*10 + '|'
    print(header_str)
    print(sep_str)

    for name, p in trials:
        if not os.path.exists(p):
            print(f"| {name:<35} | Not Found")
            continue
        res = parse_results(p)
        ppl = res['PPL']
        acc = res['Emo_Acc']
        gd1 = res['G_Dist1']
        gd2 = res['G_Dist2']
        bd1 = res['B_Dist1']
        bd2 = res['B_Dist2']
        uniq = res['Unique']
        print(f"| {name:<35} | {ppl:7.2f} | {acc:8.2f}% | {gd1:7.2f}% | {gd2:7.2f}% | {bd1:7.2f}% | {bd2:7.2f}% | {uniq:7.2f}% |")
