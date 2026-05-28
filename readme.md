# Suicidal Ideation Detection using Transformer Models
### A Comparative Study with Proposed BERT-CNN Hybrid Architecture

---

## 📌 Project Overview

This project presents a complete **NLP research pipeline** for detecting suicidal ideation in social media text (Reddit posts). It benchmarks three standard transformer models — **BERT**, **RoBERTa**, and **DistilBERT** — against a novel proposed architecture, **BERT-CNN**, which combines BERT's global contextual understanding with a Multi-Scale CNN and Attention Gate mechanism for capturing local phrase-level signals.

> **Paper Title (suggested):**
> *"Comparative Analysis of Transformer-based Models with Multi-Scale CNN Attention Fusion for Suicidal Ideation Detection in Social Media Text"*

---

## 🎯 Research Objectives

- Fine-tune and evaluate BERT, RoBERTa, and DistilBERT as baseline models on binary suicidal ideation classification
- Propose a novel **BERT-CNN hybrid** model that addresses the limitations of standard [CLS]-only classification
- Demonstrate that multi-scale local phrase features improve recall on the positive (suicide) class
- Provide a complete reproducible evaluation framework with all metrics and visualizations

---

## 📁 Project Structure

```
suicide_detection_research/
│
├── 📓 Notebooks (run in order)
│   ├── 01_setup_config.ipynb       # Hyperparameters, installs, device setup
│   ├── 02_data_eda.ipynb           # Data loading, EDA, preprocessing, DataLoaders
│   ├── 03_models.ipynb             # All model architectures (baseline + proposed)
│   ├── 04_training.ipynb           # Focal Loss, trainer, early stopping, full training
│   └── 05_evaluation.ipynb         # Metrics, plots, comparison table
│
├── 🐍 Python Scripts (equivalent to notebooks)
│   ├── config.py                   # Global configuration
│   ├── dataset.py                  # Data loading & DataLoader factory
│   ├── models.py                   # BaselineClassifier + BertCNNClassifier
│   ├── trainer.py                  # FocalLoss, EarlyStopping, train loop
│   ├── evaluate.py                 # All plots & comparison functions
│   └── main.py                     # One-command pipeline runner
│
├── data/
│   └── suicide_detection.csv       # ← Place your dataset here
│
├── outputs/                        # All plots, reports, and tables saved here
├── logs/                           # Training history JSONs
├── checkpoints/                    # Best model .pt files
└── requirements.txt
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Reddit mental health posts (NLP dataset) |
| Task | Binary classification |
| Classes | `suicide` (1) vs `non-suicide` (0) |
| Format | CSV with `text` and `class` columns |
| Split | 70% Train / 15% Validation / 15% Test (stratified) |

**CSV Format:**
```
index | text                              | class
------|-----------------------------------|------------
2     | "Ex Wife Threatening Suicide..."  | suicide
3     | "Am I weird I don't get..."       | non-suicide
```

---

## 🏗️ Model Architectures

### Baseline Models
All three baselines follow the same standard fine-tuning approach:

```
Input Text → Tokenizer → Transformer Encoder → [CLS] token → Dropout → Linear(768→2) → Logits
```

| Model | Checkpoint | Parameters | Loss |
|---|---|---|---|
| BERT | `bert-base-uncased` | 110M | Weighted CrossEntropy |
| RoBERTa | `roberta-base` | 125M | Weighted CrossEntropy |
| DistilBERT | `distilbert-base-uncased` | 66M | Weighted CrossEntropy |

---

### Proposed Model: BERT-CNN Hybrid ⭐

```
Input Text
    │
    ▼
BERT Encoder (bert-base-uncased, 12 layers)
    │
    ├── [CLS] token ──────────────────────────► (B, 768)   Global context
    │
    └── All token embeddings (B, 512, 768)
               │
               ▼
        Multi-Scale CNN  [k=2, k=3, k=4 in parallel]
               │  Bigram · Trigram · 4-gram windows
               ▼
        Attention Gate (per kernel)
               │  Learns which n-gram positions are most indicative
               ▼
        Concat → (B, 768)    Local phrase-level features
               │
               ▼
    Residual Fusion: [CLS ⊕ CNN] → LayerNorm → (B, 1536)
               │
               ▼
    MLP Head:  Linear(1536→512) → GELU → Dropout
               Linear(512→128)  → GELU → Dropout
               Linear(128→2)
               │
               ▼
    Logits (B, 2)  — Trained with Focal Loss
```

#### Key Innovations

| Innovation | Purpose |
|---|---|
| **Multi-Scale CNN** | Captures short phrase-level signals (e.g., *"want to die"*, *"end it all"*) that get diluted in long posts when using only [CLS] |
| **Attention Gate** | Learns which specific n-gram positions are most indicative of suicidal ideation, acting as a soft filter |
| **Residual Fusion** | Combines global document semantics ([CLS]) with local phrase features (CNN) for richer representation |
| **Focal Loss** | Down-weights easy non-suicide samples, focuses training on hard misclassified cases — better for class imbalance |

---

## 📐 Evaluation Metrics

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness — baseline sanity check |
| **Precision** | Of all predicted-suicide posts, how many truly were? Reduces false alarms |
| **Recall** | Of all actual suicide posts, how many were caught? Most clinically critical |
| **Macro F1** | Balances precision and recall equally across both classes — **primary model selection metric** |
| **F1 (Suicide)** | F1 computed only for the positive class — most clinically meaningful single number |
| **ROC-AUC** | Threshold-independent separation ability — best for comparing models under imbalance |

> **Why Recall is the most important metric:** In a mental health context, a False Negative (missing a genuinely suicidal post) is far more dangerous than a False Positive. The BERT-CNN model is specifically designed to improve Recall on the suicide class.

---

## 📈 Expected Results

| Model | Accuracy | Macro F1 | F1-Suicide | ROC-AUC |
|---|---|---|---|---|
| BERT | ~0.92 | ~0.91 | ~0.90 | ~0.96 |
| RoBERTa | ~0.93 | ~0.92 | ~0.91 | ~0.97 |
| DistilBERT | ~0.91 | ~0.90 | ~0.89 | ~0.95 |
| **BERT-CNN** | **~0.94+** | **~0.93+** | **~0.93+** | **~0.97+** |

*Actual values depend on your full dataset size and class distribution.*

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Max sequence length | 512 |
| Batch size | 16 |
| Epochs | 5 (with early stopping) |
| Learning rate (encoder) | 2e-5 |
| Learning rate (classifier head) | 2e-4 (10×) |
| Warmup ratio | 10% |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Early stopping patience | 3 epochs |
| CNN filters | 256 |
| CNN kernel sizes | 2, 3, 4 |
| Focal Loss γ | 2.0 |
| Focal Loss α | 0.25 |
| Dropout | 0.3 |

---

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset
```
data/suicide_detection.csv
```

### 3a. Run via Jupyter Notebooks (recommended for research)
Open and run notebooks in order:
```
01_setup_config.ipynb  →  02_data_eda.ipynb  →  03_models.ipynb
→  04_training.ipynb  →  05_evaluation.ipynb
```

### 3b. Run via Python Script (faster, one command)
```bash
# Train all 4 models
python main.py

# Train specific models only
python main.py --models BERT RoBERTa

# Only run evaluation (requires saved checkpoints)
python main.py --eval-only

# Custom dataset path
python main.py --data path/to/your_data.csv
```

---

## 📤 Outputs

After running, the `outputs/` directory will contain:

| File | Description |
|---|---|
| `cm_{MODEL}.png` | Confusion matrix per model |
| `history_{MODEL}.png` | Train/val Loss & F1 curves per epoch |
| `roc_auc_comparison.png` | All models' ROC curves on one plot |
| `metric_comparison.png` | Side-by-side bar chart of all metrics |
| `comparison_table.csv` | Final results table (import into your thesis) |
| `report_{MODEL}.txt` | Full classification report per model |
| `results_{MODEL}.json` | Raw metrics dictionary |

---

## 🔬 Why CNN Helps (Technical Explanation)

Standard BERT [CLS]-only classification is limited for long posts:

```
Plain BERT:
"[300 words of general distress] ... want to die ... [50 more words]"
       ↓
[CLS] summarizes ALL 512 tokens equally
       ↓
"want to die" at token 340 contributes very little weight
       ↓
Model predicts: non-suicide ← FALSE NEGATIVE
```

```
BERT-CNN:
Same post
       ↓
CNN kernel=3 scans every 3-token window across all 512 positions
       ↓
Hits "want to die" at position 340 → HIGH local activation
       ↓
Attention gate amplifies this signal
       ↓
Fused with [CLS]: model has BOTH global context AND local signal
       ↓
Model predicts: suicide ← CORRECT
```

The CNN captures **short-term local context** while BERT handles **long-term global context**. Together they are stronger than either alone.

---

## 📚 References

1. Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
2. Liu et al. (2019) — *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
3. Sanh et al. (2019) — *DistilBERT, a distilled version of BERT*
4. Lin et al. (2017) — *Focal Loss for Dense Object Detection*
5. Kim (2014) — *Convolutional Neural Networks for Sentence Classification*

---

## ⚠️ Ethical Note

This project is strictly for **academic research purposes**. The dataset contains sensitive mental health content. All work should be conducted responsibly with respect to the individuals whose posts form the dataset, and any real-world deployment of such a system should be supervised by qualified mental health professionals.
