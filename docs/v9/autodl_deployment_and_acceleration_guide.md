# AutoDL 云端部署与极速传输全流程操作手册

> **文档适用对象**：CASE-EPCL (V9-Clean) 云端 RTX 4090D (24GB) 生产级训练  
> **核心目标**：彻底解决网页端上传慢、容易断流、文件体积庞大的问题，实现极速上机、秒级健康自检与后台稳健训练。

---

## 一、为什么网页端上传慢？（体积解剖）

整个工程在本地若全选打包会高达 **16GB**，其中绝大多数是历史冗余文件：

| 目录/文件 | 实际体积 | 云端必要性 | 解决策略 |
| :--- | :--- | :--- | :--- |
| **`save/`** | **14.04 GB** | ❌ 坚决不传 | 本地历史权重，云端训练从零启动，绝对排除 |
| **`vectors/`** | **990 MB** | ✅ 必须 | **痛点大头**。可在云端千兆机房直连下载，**无需从本地上传** |
| **`data/`** | **893 MB**（压缩后约 260MB） | ✅ 必须 | 预处理好的数据集，只需上传压缩后的精简包 |
| **代码与脚本** (`src/`, `utils/`, `tests/`, `main.*`, `eval.sh`) | **不足 1 MB** | ✅ 必须 | 秒级同步 |

---

## 二、三套传输加速方案（按推荐度排序）

### 方案 A：分离极速法（强烈推荐，本地只需传 260MB）
利用云端机房的千兆带宽直接拉取 1GB 的 GloVe 词向量，本地只上传代码与数据集。

#### 1. 本地打包精简包（仅代码 + 数据集，约 260MB）
在本地 Windows PowerShell 终端中执行：
```powershell
Compress-Archive -Path data, src, utils, tests, main.py, main.sh, eval.sh, requirements.txt -DestinationPath CASE_light.zip -Force
```
*生成的 `CASE_light.zip` 仅 200 多兆，在 JupyterLab 网页端上传仅需 1~2 分钟。*

#### 2. 上传并解压
在 AutoDL 的 JupyterLab 中将 `CASE_light.zip` 上传至 `/root/autodl-tmp/`，终端解压：
```bash
cd /root/autodl-tmp
apt-get update && apt-get install -y unzip wget
unzip CASE.zip -d /root/autodl-tmp/CASE
cd /root/autodl-tmp/CASE
```

#### 3. 云端千兆带宽极速下载 GloVe（约 30 秒）
进入 `CASE` 目录，开启学术加速直连下载词向量，无需耗费本地上行带宽：
```bash
# 开启学术网络加速
source /etc/network_turbo

# 创建 vectors 目录并进入
mkdir -p vectors && cd vectors

# 从斯坦福官方极速下载（学术加速下约 30-50MB/s）
wget https://nlp.stanford.edu/data/glove.6B.zip

# 解压出 300d 词向量
unzip glove.6B.zip glove.6B.300d.txt

# 删除多余压缩包，节省空间
rm glove.6B.zip

# 回到项目根目录
cd /root/autodl-tmp/CASE
```

---

### 方案 B：专业 SFTP 工具传输（支持断点续传、满速上传）
如果不愿意拆分文件，推荐使用专业 SFTP 客户端（**Xftp / WinSCP / FileZilla / MobaXterm**）：
1. 在 AutoDL 控制台实例列表中，查看该实例的 **“SSH登录指令”** 和 **“密码”**（例如：`ssh -p 35212 root@connect.westb.seetacloud.com`）。
2. 在 WinSCP / Xftp 中新建连接：
   - **主机名/Host**：`connect.westb.seetacloud.com`（以您控制台显示的为准）
   - **端口号/Port**：`35212`（以您控制台显示的为准）
   - **用户名**：`root`
   - **密码**：您的实例密码
3. 连接成功后，直接将本地打包好的 `CASE_deploy.zip` 拖拽到右侧的 `/root/autodl-tmp/` 目录下。专业 SFTP 协议不受浏览器上传缓存限制，速度可达本地宽带上限。

---

### 方案 C：AutoDL“网盘助手”（阿里云盘 / 百度网盘免流同步）
1. 将本地打包的文件上传至您的个人 **阿里云盘**。
2. 打开 AutoDL JupyterLab，点击左侧菜单栏的 **【网盘助手】** 图标。
3. 扫码授权阿里云盘，选择文件一键转存到 `/root/autodl-tmp/`，云对云千兆直连，通常秒级完成。

---

## 三、云端环境初始化与依赖安装

进入 `/root/autodl-tmp/CASE` 目录后，执行以下命令：

```bash
# 1. 确认学术加速已开启
source /etc/network_turbo

# 2. 安装项目依赖（已剔除冗余包，秒级装完）
pip install -r requirements.txt

# 3. 下载 NLTK 分词与语料依赖包（加入代理信任参数，避免被拦截）
NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. 执行开机 15 项全覆盖健康检查
python tests/test_sanity.py
```

**预期输出**：
```text
................
----------------------------------------------------------------------
Ran 15 tests in 0.00x s

OK
```
*看到 OK 表明：PyTorch 2.1.0、CUDA 12.1、4090D GPU 算子和 V9-Clean 架构全部处于健康待命状态。*

---

## 四、后台持久化训练（防断网终止）

云端训练严禁在裸前台终端直接运行，否则一旦浏览器标签卡死或网络波动，训练进程会被系统信号直接杀死。

### 1. 启动持久会话
```bash
# 创建一个名为 train 的后台持久会话
tmux new -s train
```

### 2. 启动 24G 满血训练
在 tmux 窗口内输入：
```bash
bash main.sh 24G
```
*(系统将以 batch_size=16, accum_steps=1, fp32 全精度无损启动生产级训练)*

### 3. 脱离后台（可以安心关闭网页或电脑）
- 键盘同时按下：`Ctrl` + `B`
- 松开后，立刻按一下字母键：`D` (Detach)
- 此时控制台提示 `[detached (from session train)]`，代表训练已稳妥在后台运行，此时关闭电脑完全不受影响。

### 4. 重新连入查看进度
随时在终端输入：
```bash
tmux attach -t train
```
即可重新切入实时训练界面。

---

## 五、实时监控与结果验证

### 1. 显卡监控（新开一个终端窗口）
```bash
watch -n 1 nvidia-smi
```
- **显存占用**：约 6GB ~ 10GB（24GB 显存绰绰有余，零 OOM 风险）；
- **GPU 利用率**：通常在 85% ~ 98% 之间高效运转。

### 2. 实时跟踪训练日志
```bash
tail -f logs/train.log
```

### 3. 训练完成后一键全指标度量
训练收敛后，在 `/root/autodl-tmp/CASE` 下执行：
```bash
bash eval.sh
```
该脚本将全自动加载最优 checkpoint，一键输出：
- 生成质量：PPL, BLEU-1/2/3/4, Distinct-1/2, Unique-1/2
- 情绪识别：Emotion Accuracy
- 流形几何：Alignment, Uniformity, DBI (戴维森堡丁指数)
- 混淆矩阵与 t-SNE 可视化图表
