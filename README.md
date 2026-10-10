# CIFAR-100 / VGG16 ANN 训练

专门用于同学下载、复现及改进 ANN 的仓库。包含完整 CIFAR-100 数据分块，不包含 SNN 代码、模型权重或私人凭据。

## 当前最佳已验证配置

- 官方测试集 Top1 **76.16%**、Top5 **93.43%**，10,000 张图片；训练 seed=42。
- 300 轮，按验证集选中第 297 轮。最佳训练记录验证 Top1=77.06%，不是测试准确率。
- VGG16-BN：13 个卷积层、3 个全连接层，普通 ReLU、MaxPool，分类器 512→4096→4096→100。
- RandomCrop、水平翻转、AutoAugment（torchvision CIFAR10 策略用于 CIFAR-100）、Mixup α=0.2。
- SGD、Nesterov、学习率 0.1、momentum 0.9、weight decay 5e-4，5 轮 warmup 后 cosine 至 1e-5，batch size=128。
- 官方训练集分为 45,000 训练 / 5,000 分层验证；官方 10,000 测试图片不参与训练、校准或选权重。

“最佳”指本项目已完成并核验的版本，不代表所有超参数中的全局最优；不同软件、GPU 或随机种子可能得到不同精度。
七个核心源文件与产生 76.16% 结果的源码快照 SHA256 完全一致，记录见 `reference/source_sha256.json`。默认配置使用该版本；仅使用本机相对数据和输出路径。

## Linux / 服务器下载和安装

```bash
git clone https://github.com/zyh3592/cifar100-vgg16-ann-training.git
cd cifar100-vgg16-ann-training
conda create -n cifar100-ann python=3.10 -y
conda activate cifar100-ann
# 复现原 Ubuntu 22.04 / NVIDIA driver 535 环境：
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 'Pillow>=10,<13'
python -B check_environment.py --help
python -B prepare_data.py
```

公开仓库 clone/pull 不需要登录 GitHub。其他显卡/驱动请安装匹配的官方 PyTorch 与 torchvision 组合；不要盲目混装版本。`requirements-server.txt` 记录原服务器版本。
Windows 用户请将项目放在 `D:\Codex_WorkSpace` 内，代码会把临时文件和输出限制在该工作区，禁止写入 C 盘。

## 一键训练并测试

```bash
python -B run_ann.py
```

自动生成带 UTC 时间和随机后缀的独立输出目录，训练结束后用最佳验证权重评估一次官方测试集。
可指定新目录或 seed；已有目录会拒绝覆盖：

```bash
python -B run_ann.py --output-dir output/classmate_seed42_01 --seed 42 --device cuda:0
```

PyCharm：脚本设为 `run_ann.py`，工作目录设为仓库根目录，解释器选择 `cifar100-ann` 环境；参数可留空。
直接使用 `train.py` 时，仅支持训练参数；`--run` 和 `--run-dir` 不是它的参数。

## 断点续训、单独测试

```bash
python -B train.py --config configs/vgg16_cifar100.json --output-dir output/classmate_seed42_01 --resume output/classmate_seed42_01/last.pt --device cuda:0
python -B evaluate.py --checkpoint output/classmate_seed42_01/best.pt --output output/classmate_seed42_01/official_test_metrics.json
```

续训必须使用原目录、原 seed 和原训练设置（若开始时改过 seed，续训也要加同样的 `--seed`）；源码改变会拒绝续训，应建立新实验。
`evaluate.py` 是显式测试工具：同一路径重复调用会替换指标文件，保留旧评估时请指定不同 `--output`。

## 如何合作提高精度且兼容后续 SNN

优先比较学习率、训练轮数、weight decay、Mixup/CutMix、增强策略和随机种子。每次复制 JSON 为新配置，并分配新输出目录。
为继续使用现有 SNN 模块，请保持 `model.py`、普通 ReLU、BN/MaxPool、hidden_dim=4096、类别顺序及归一化不变，不要自行改成 QCFS 或其他激活。
修改核心源码会产生新的哈希，这是新实验的来源记录，不能继续冒用原 76.16% 的结果。
使用验证集比较改进和选择权重，再对最终配置报告官方测试集；避免反复按测试集筛选超参数。

每个实验保留：`best.pt`、`last.pt`、`history.csv`、`summary.json`、`config.json`、`split_indices.json`、`environment.json`、`source/`、`source_sha256.json`、`official_test_metrics.json`。
若同学另行修改代码，也请保存修改后的配置与源文件。结果包回传后可接入原 SNN 项目；此仓库只负责 ANN。

## 数据来源、完整性和检查

数据来自 [CIFAR 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)，作者 Alex Krizhevsky、Vinod Nair、Geoffrey Hinton。
完整官方 python 压缩包 169,001,437 字节，分为 5 块；普通 Git 下载即可，无需 Git LFS。
`prepare_data.py` 合并各块，校验分块 SHA256、整体 SHA256/官方 MD5，并检查 train/test/meta 的官方 MD5，再安全解压到 `data/raw`。
原始数据的使用权利与条件归原作者；本仓库不对其重新授权。

```bash
python -B -m unittest discover -s tests -v
```

`reference/` 只保存原最佳实验的指标、环境和来源摘要，便于对照，不含训练权重。
