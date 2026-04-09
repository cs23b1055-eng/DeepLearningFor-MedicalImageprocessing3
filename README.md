Here's the README text you can copy:

---

# Embryo Stage Classification — 5 CNN Backbones + Custom OrdinalFocalLoss

## Overview
This project classifies embryo developmental stages from Time-Lapse Imaging (TLI) videos using five pretrained CNN backbones with a custom ordinal-aware focal loss function.

**Dataset:** Gomez et al. (2022) — 704 TLI videos, 2.4M images, 16 morphokinetic phases
**Source:** [zenodo.org/records/6390798](https://zenodo.org/records/6390798)

---

## Models Used
| Backbone | Params | Strategy |
|---|---|---|
| MobileNetV2 | 3.4M | Freeze 3 ep → unfreeze 2 ep |
| InceptionV3 | 27M | Freeze 3 ep → unfreeze 2 ep |
| VGG16 | 138M | Head-only training (frozen backbone) |
| VGG19 | 144M | Head-only training (frozen backbone) |
| ResNet50 | 25M | Freeze 3 ep → unfreeze 2 ep |

---

## Custom Loss Function — `OrdinalFocalLoss`

Standard cross-entropy treats all misclassifications equally. For embryo staging, predicting stage *t5* instead of *t6* is far less harmful than predicting *tHB* instead of *tPB2*. The custom `OrdinalFocalLoss` addresses this:

**Formula:**
```
L = α_t · (1 − p_t)^γ · (CE_smooth + λ · Ord)
```

| Component | Role |
|---|---|
| `α_t` | Per-class inverse-frequency weights → handles class imbalance |
| `(1−p_t)^γ` | Focal modulator → down-weights easy/confident predictions |
| `CE_smooth` | Label-smoothed cross-entropy → improves calibration |
| `Ord` | Ordinal penalty — penalises large stage-index jumps more |

### 7 Desirable Properties (all verified in Section 5 of notebook)
1. **Non-negativity** — Loss ≥ 0 always
2. **Zero at perfection** — Loss → 0 when predictions are perfect
3. **Ordinal awareness** — Distant stage errors cost more than adjacent ones
4. **Imbalance handling** — Rare stages up-weighted via inverse-frequency α
5. **Smoothness / differentiability** — All ops are differentiable; gradients are finite
6. **Calibration** — Label smoothing (ε=0.1) prevents overconfidence
7. **Focal property** — Hard/ambiguous samples receive higher loss weight

---

## Training Strategy
- `FRAME_STRIDE = 5` → only 20% of frames (~480K instead of 2.4M)
- `EPOCHS = 5` per model with frozen backbone
- `BATCH_SIZE = 4` with gradient accumulation ×4 (effective batch = 16)
- AMP (float16) halves VRAM on GPU
- Images resized to **128px** (160px for Inception)
- Sequential training: load → train → save → free memory

---

## 16 Morphokinetic Phases
`tPB2 · tPNa · tPNf · t2 · t3 · t4 · t5 · t6 · t7 · t8 · t9+ · tM · tSB · tB · tEB · tHB`

---

## Outputs
| File | Description |
|---|---|
| `embryo_classifier_<backbone>.pth` | Saved model checkpoint |
| `history_<backbone>.json` | Training/val loss, accuracy, MAE per epoch |
| `curves_all.png` | Training curves for all backbones |
| `confusion_all.png` | Normalised confusion matrices |
| `comparison.png` | Final accuracy & MAE bar charts |

---

## Requirements
```
torch torchvision pandas Pillow tqdm matplotlib seaborn scikit-learn
```

---


---

*Roll No: CS23B1055*
