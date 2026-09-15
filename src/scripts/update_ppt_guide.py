import os

ppt_path = r"C:\Users\chy\Desktop\新建文件夹 (3)\板块3_导师汇报开题PPT参考指南.md"

if os.path.exists(ppt_path):
    with open(ppt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    target = """### 2. 【版面构图建议】
* **核心配图（强烈推荐直接插入本图）**：直接插入高清架构演化全景图 **`images/fig5_case_vs_epcl_architecture.png`**（原版 CASE 与改进后 CASE-EPCL 架构演进与 4 处手术式改进全景对比）；
* **版面布局**：左侧展示原版 CASE 数据流与缺陷（MIM局部坍缩、混合层分类导致梯度倒灌）；右侧展示 CASE-EPCL 4 处手术式改进（EPCL超球面质心、物理通道解耦、MoP-DR动态路由、28k步时序冻结退火）。"""

    replacement = """### 2. 【版面构图建议】
* **核心配图（二选一或分栏展示）**：
  * **方案 A（全景宏观对比）**：插入 **`images/fig5_case_vs_epcl_architecture.png`**（原版 CASE vs CASE-EPCL 架构演进与 4 处手术式改进全景对比）；
  * **方案 B（极简挂载特写·强烈推荐导师汇报）**：插入 **`images/fig6_epcl_mounting_schematic.png`**（原版 CASE 骨干网络与 EPCL 模块即插即用挂载拓扑图，一眼看清 EPCL 仅仅外挂在 fine_emotion 节点上，主干生成毫无破坏）；
* **版面布局**：主图置于画面中央偏右，左侧保留 3 条大字核心立论，讲解时指着挂载线强调“主干生成完好无损、旁路外挂规整空间”。"""

    if target in text:
        text = text.replace(target, replacement)
        with open(ppt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Updated PPT guide with Fig 6 successfully!")
    else:
        print("Target string not found in PPT guide!")
