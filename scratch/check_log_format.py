import re

def extract_dev_trajectory(log_file):
    steps = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 寻找形如: step: 499, dev ppl: 67.14... 或者 EVAL valid
                # 在本项目中, train_one_batch / main.py 打印格式通常为:
                # EVAL valid ... 或 step: ...
                if 'EVAL\tBOW_Loss' in line or 'test\t' in line:
                    continue
                # 匹配 valid 行: valid\t5.123\t...
                m = re.search(r'valid\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)', line)
                # 或者标准打印
                # 让我们先看看日志中的真实打印格式
    except Exception as e:
        print(f'Error reading {log_file}: {e}')
    return steps

# 读前几行包含 valid 或 ppl 的行
with open('results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'valid' in line.lower() and ('ppl' in line.lower() or '\t' in line):
            print('Sample line:', line.strip())
            break
