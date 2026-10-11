# PepEmbed

PepEmbed 将一条以 SMILES 表示的肽转换成固定维度的图级嵌入。它把分子按化学结构拆解为
片段，用预训练的分子编码器对每个片段编码，再用消息传递网络配合注意力池化聚合出片段图，
得到图级表示 `h_G`。

提供三个预训练变体。它们采用相同的流程，区别主要在于片段的初始编码方式：

| 方法 | 片段初始编码 | 片段维度 |
|---|---|---|
| `molformer` | MoLFormer `[CLS]` 池化 | 768 |
| `molt5` | MolT5 编码器 mask-mean 池化 | 768 |
| `moler` | MoLeR `z(512)` + 手性 `c(8)` + 残基类型 `t(6)` | 526 |

用 `--method {molformer,molt5,moler}` 选择变体。每个变体的编码后端、基座模型目录、预训练权重
和超参数都定义在 `config.yaml` 中。

> 本文档的英文版见 [`README.md`](README.md)。

## 目录结构

```
.
├── get_embedding.py       # 统一入口
├── config.yaml            # 方法 → 目录/权重/超参数 映射（相对路径）
├── requirements.txt       # 安装文件（见“安装”一节）
├── split_fragment.py      # 片段分解（所有方法共用）
├── common_sets.json       # 常见残基/连接子集合（所有方法共用）
├── common/                # 共享的模型/数据代码（FragmentModel 等）
└── model/
    ├── molformer/{backend.py, best.pth, base_model/}
    ├── molt5/{backend.py, best.pth, base_model/}
    └── moler/{backend.py, moler_526.py, best.pth, base_model/}
```

- `best.pth`：本项目预训练得到的 `FragmentModel` 权重。
- `base_model/`：各方法所需的基座模型（MoLFormer-XL / molt5-base / MoLeR）；
  其包含的文件见第一节。

## 一、准备基座模型（需自行下载）

`model/<method>/base_model` 必须是可被 `transformers` / `molecule_generation` 直接加载的
模型目录。把下列文件下载到 `model/<method>/base_model/`。

### 1. molformer → `model/molformer/base_model/`
从 <https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/tree/main> 下载以下文件：
```
config.json
configuration_molformer.py
modeling_molformer.py
model.safetensors
tokenizer.json
tokenizer_config.json
```

### 2. molt5 → `model/molt5/base_model/`
从 <https://huggingface.co/laituan245/molt5-base/tree/main> 下载以下文件：
```
config.json
pytorch_model.bin
spiece.model
tokenizer_config.json
```

### 3. moler → `model/moler/base_model/`
从 <https://figshare.com/ndownloader/files/34642724> 下载预训练的 MoLeR 检查点：
```
GNN_Edge_MLP_MoLeR__20220224_071623_best.pkl
```

## 二、安装

安装下列软件包后即可运行模型。

| 软件包 | 需要它的方法 | 说明 |
|---|---|---|
| `torch` | 全部 | |
| `rdkit` | 全部 | |
| `transformers` | `molformer`、`molt5` | |
| `PyYAML` | 全部 | 解析配置 |
| `numpy` | 全部 | |
| `lmdb` | 全部 | 共享的 `common` 包会 import |
| `tensorflow` | 仅 `moler` | MoLeR 推理在 CPU 上运行 |
| `molecule_generation` | 仅 `moler` | 不在 PyPI 上，需从源码安装 |

一键安装上述全部依赖（已在 Python 3.9 上验证）：

```bash
pip install -r requirements.txt
```

`molecule_generation` 未发布到 PyPI，因此使用 `moler` 方法时还需从官方仓库安装：

```bash
git clone https://github.com/google-deepmind/molecule_generation
pip install -e molecule_generation
```

## 三、用法

```bash
# 单条 SMILES（方法默认取 config.yaml 的 default_method）
python get_embedding.py "C(C(=O)NCC(=O)O)N"
python get_embedding.py "C(C(=O)NCC(=O)O)N" --method moler --output all

# CSV 批量（务必给 --save）
python get_embedding.py --csv input.csv --method molt5 \
    --output h_G,alpha,fragment_meta --save embeddings.pt
```

### 常用参数

| 参数 | 说明 |
|---|---|
| `--method {molformer,molt5,moler}` | 选择片段初始编码方式（默认 `config.yaml:default_method`） |
| `--config` | config 路径（默认 `config.yaml`） |
| `--output` | 输出键：`h_G,node_repr,alpha,weighted_node_repr,fragment_meta` 或 `all` |
| `--csv` / `--save` | CSV 批量输入 / 结果 `.pt` 保存路径 |
| `--model-dir` / `--ckpt` / `--common-sets` | 覆盖 config 中的基座 / 权重 / 公共集合路径 |
| `--batch-size` / `--workers` | 覆盖片段编码批大小 / MoLeR worker 数 |
| `--gnn-batch-size` | GNN 前向批大小（CSV 模式） |
| `--device` | `cpu` / `cuda`（默认自动探测） |
