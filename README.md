# PepEmbed

PepEmbed turns a peptide, given as SMILES, into a fixed-size graph-level embedding.
It first decomposes the molecule into chemically meaningful fragments, encodes every
fragment with a pretrained molecular encoder, and then aggregates the resulting fragment
graph with a message-passing network plus attention pooling to produce the final
representation `h_G`.

Three pretrained variants are provided. They share an identical pipeline and differ only
in the fragment initial encoding:

| Method | Fragment initial encoding | Fragment dim |
|---|---|---|
| `molformer` | MoLFormer `[CLS]` pooling | 768 |
| `molt5` | MolT5 encoder mask-mean pooling | 768 |
| `moler` | MoLeR `z(512)` + chirality `c(8)` + residue kind `t(6)` | 526 |

Select a variant with `--method {molformer,molt5,moler}`. Each variant's encoding backend,
base-model directory, pretrained weights and hyperparameters are defined in `config.yaml`.

> A Chinese version of this document is kept as [`README_zh.md`](README_zh.md).

## Directory layout

```
.
├── get_embedding.py       # unified entry
├── config.yaml            # method -> directory / weights / hyperparameters (relative paths)
├── requirements.txt       # install file (see Installation)
├── split_fragment.py      # fragment decomposition (shared by all methods)
├── common_sets.json       # common residue/linker sets (shared by all methods)
├── common/                # shared model/data code (FragmentModel etc.)
└── model/
    ├── molformer/{backend.py, best.pth, base_model/}
    ├── molt5/{backend.py, best.pth, base_model/}
    └── moler/{backend.py, moler_526.py, best.pth, base_model/}
```

- `best.pth`: the `FragmentModel` weights pretrained by this project.
- `base_model/`: the base model required by each method (MoLFormer-XL / molt5-base / MoLeR);
  see section 1 for the files it contains.

## 1. Prepare the base models (download them yourself)

`model/<method>/base_model` must be a directory that `transformers` / `molecule_generation`
can load directly. Download the files listed below into `model/<method>/base_model/`.

### 1. molformer -> `model/molformer/base_model/`
Download these files from
<https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/tree/main>:
```
config.json
configuration_molformer.py
modeling_molformer.py
model.safetensors
tokenizer.json
tokenizer_config.json
```

### 2. molt5 -> `model/molt5/base_model/`
Download these files from
<https://huggingface.co/laituan245/molt5-base/tree/main>:
```
config.json
pytorch_model.bin
spiece.model
tokenizer_config.json
```

### 3. moler -> `model/moler/base_model/`
Download the pretrained MoLeR checkpoint from
<https://figshare.com/ndownloader/files/34642724>:
```
GNN_Edge_MLP_MoLeR__20220224_071623_best.pkl
```

## 2. Installation

Install the packages below, then run the model.

| Package | Needed by | Notes |
|---|---|---|
| `torch` | all | |
| `rdkit` | all | |
| `transformers` | `molformer`, `molt5` | |
| `PyYAML` | all | config parsing |
| `numpy` | all | |
| `lmdb` | all | imported by the shared `common` package |
| `tensorflow` | `moler` only | MoLeR inference runs on CPU |
| `molecule_generation` | `moler` only | not on PyPI, install from source |

One-click install of everything above (tested on Python 3.9):

```bash
pip install -r requirements.txt
```

`molecule_generation` is not published on PyPI, so for the `moler` method install it
from the official repository:

```bash
git clone https://github.com/google-deepmind/molecule_generation
pip install -e molecule_generation
```

## 3. Usage

```bash
# Single SMILES (method defaults to config.yaml's default_method)
python get_embedding.py "C(C(=O)NCC(=O)O)N"
python get_embedding.py "C(C(=O)NCC(=O)O)N" --method moler --output all

# CSV batch (must pass --save)
python get_embedding.py --csv input.csv --method molt5 \
    --output h_G,alpha,fragment_meta --save embeddings.pt
```

### Common arguments

| Argument | Description |
|---|---|
| `--method {molformer,molt5,moler}` | Select the fragment initial encoding (default: `config.yaml:default_method`) |
| `--config` | config path (default: `config.yaml`) |
| `--output` | Output keys: `h_G,node_repr,alpha,weighted_node_repr,fragment_meta`, or `all` |
| `--csv` / `--save` | CSV batch input / `.pt` save path for results |
| `--model-dir` / `--ckpt` / `--common-sets` | Override the base model / weights / common sets path from config |
| `--batch-size` / `--workers` | Override the fragment-encoding batch size / MoLeR worker count |
| `--gnn-batch-size` | GNN forward batch size (CSV mode) |
| `--device` | `cpu` / `cuda` (auto-detected by default) |

### About the config schema

Every method entry in `config.yaml` carries the same keys. The `batch_size` and `workers`
keys exist on all three for uniformity, but each backend only honors the one it uses:
`molformer` / `molt5` encode on the GPU in batches (use `batch_size`), while `moler`
encodes on the CPU through a worker pool (use `workers`). The key a backend does not use
is still forwarded but ignored (backends accept `**opts`), so it is harmless.
