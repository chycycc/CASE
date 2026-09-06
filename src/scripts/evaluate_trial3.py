import os
import re
import sys
from nltk import word_tokenize

def calc_distinct_n(n, candidates):
    d = set()
    total = 0
    tokenized = [word_tokenize(c) for c in candidates]
    for s in tokenized:
        for i in range(len(s) - n + 1):
            key = tuple(s[i : i + n])
            d.add(key)
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
            
    g_d1 = calc_distinct_n(1, greedies)
    g_d2 = calc_distinct_n(2, greedies)
    b_d1 = calc_distinct_n(1, beams)
    b_d2 = calc_distinct_n(2, beams)
    
    unique_count = len(set(greedies))
    total_count = len(greedies)
    unique_ratio = unique_count / (total_count + 1e-16) * 100
    
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
        'Unique_Ratio': unique_ratio,
        'Unique_Count': unique_count,
        'Total_Count': total_count
    }

def get_best_val_loss(log_path):
    if not os.path.exists(log_path):
        return None
    min_loss = 999.0
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 常见格式: best loss: 37.9531 或 validation loss 或 eval 结果
            m = re.search(r'best\s+loss[:=]\s*([0-9\.]+)', line, re.I)
            if m:
                val = float(m.group(1))
                if val < min_loss:
                    min_loss = val
            m2 = re.search(r'loss:\s*([0-9\.]+).*best model saved', line, re.I)
            if m2:
                val = float(m2.group(1))
                if val < min_loss:
                    min_loss = val
    return min_loss if min_loss < 990.0 else None

if __name__ == '__main__':
    experiments = [
        ('V5 Trial 1 (div=2.0x, 基准)', 'save/epcl_v5_trial1/results.txt', 'train_v5_trial1.log'),
        ('V5 Trial 3a (div=1.8x)', 'save/epcl_v5_trial3a/results.txt', 'train_v5_trial3a.log'),
        ('V5 Trial 3b (div=2.2x)', 'save/epcl_v5_trial3b/results.txt', 'train_v5_trial3b.log'),
        ('V5 Trial 3c (div=2.5x)', 'save/epcl_v5_trial3c/results.txt', 'train_v5_trial3c.log'),
    ]

    print(f"| {'实验版本':<24} | {'div':<6} | {'Emo Acc%':<9} | {'G-Dist1%':<9} | {'G-Dist2%':<9} | {'B-Dist1%':<9} | {'B-Dist2%':<9} | {'PPL':<7} | {'唯一率%':<8} |")
    print("|" + "-"*26 + "|" + "-"*8 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*9 + "|" + "-"*10 + "|")

    div_weights = ['2.0x', '1.8x', '2.2x', '2.5x']
    for (name, r_path, l_path), dw in zip(experiments, div_weights):
        if not os.path.exists(r_path):
            print(f"| {name:<24} | {dw:<6} | 文件不存在: {r_path}")
            continue
        res = parse_results(r_path)
        print(f"| {name:<24} | {dw:<6} | {res['Emo_Acc']:9.2f}% | {res['G_Dist1']:9.2f}% | {res['G_Dist2']:9.2f}% | {res['B_Dist1']:9.2f}% | {res['B_Dist2']:9.2f}% | {res['PPL']:7.2f} | {res['Unique_Ratio']:8.2f}% |")

