# 🔬 Patch-Range Transformer (PRT)
### Efficient Local Attention for Vision Transformers

---

## 📌 Overview

Vision Transformers (ViT) have demonstrated strong performance in image classification by leveraging global self-attention. However, full pairwise attention leads to **quadratic computational complexity O(N²)** and weak spatial inductive bias, making ViT inefficient for small to medium-scale datasets.

This project introduces the **Patch-Range Transformer (PRT)**, a simple yet effective modification that enables **spatially-constrained local attention** without increasing model complexity.

PRT restricts each patch to attend only to its neighboring patches within a fixed **Chebyshev distance (R)** while a global **CLS token** preserves long-range dependencies.

---

## ⚙️ Key Idea

- Local attention using spatial neighborhood (Chebyshev distance)
- Global CLS token for long-range communication
- No additional parameters
- Drop-in replacement for standard Vision Transformer

---

## 🚀 Key Contributions

- ✔️ Novel **Patch-Range attention mechanism**
- ✔️ Efficient local attention without window shifting or hierarchy
- ✔️ **Up to 61% reduction in attention FLOPs**
- ✔️ Same parameter count as standard ViT
- ✔️ Simple, scalable, and efficient design

---

## 📊 Experimental Results

All models trained **from scratch (no pretrained weights)**

### CIFAR-10
| Model | Accuracy | Params | FLOPs |
|------|---------|--------|--------|
| **PRT (Ours)** | **82.11%** | 2.69M | 0.000307 |
| ViT | 81.11% | 2.69M | 0.000786 |
| Swin | 84.66% | 27.5M | 0.000461 |

---

### CIFAR-100
| Model | Accuracy |
|------|---------|
| **PRT (Ours)** | **54.59%** |
| ViT | 52.43% |
| Swin | 51.77% |

---

### Caltech-101
| Model | Accuracy |
|------|---------|
| **PRT (Ours)** | **53.97%** |
| ViT | 49.83% |
| Swin | 50.52% |

---

## 📈 Ablation Study (Attention Range R)

| R | Patches Attended | Accuracy |
|--|------------------|----------|
| 1 | 9  | 53.57 |
| 2 | 25 | **53.97** |
| 3 | 49 | 53.63 |
| 4 | 81 | 52.59 |

👉 **R = 2 gives the best trade-off between efficiency and accuracy**

---

## 🏗️ Architecture

- Transformer Blocks: 6  
- Embedding Dimension: 192  
- Attention Heads: 6  
- Patch Size: 4×4 (CIFAR), 8×8 (Caltech-101)  
- Parameters: ~2.69M  


---
## 📂 Repository Structure

```
.
├── prt.py     # Patch-Range Transformer (PRT)
├── vit.py     # Standard Vision Transformer
├── swin.py    # Swin Transformer (timm-based)
```

---

## 🧪 Usage

### Install dependencies
```bash
pip install torch timm
```

### Example
```python
from prt import PatchRangeTransformer

model = PatchRangeTransformer(
    img_size=32,
    patch_size=4,
    num_classes=10,
    embed_dim=192,
    depth=6,
    num_heads=6,
    R=2
)
```

---

## 🔗 Experiments (Kaggle)

### CIFAR-10  
https://www.kaggle.com/code/balamuruganaiml/patch-range-transformer-prt-cifar-10  

### CIFAR-100  
https://www.kaggle.com/code/balamuruganaiml/patch-range-transformer-prt-cifar-100  

### Caltech-101  
https://www.kaggle.com/code/kbalamurugank/patch-range-transformer-prt-caltech-101  

---

## 📄 Paper

https://doi.org/10.5281/zenodo.19273585  

---

## 🧠 Citation

```
Bala Murugan. Patch-Range Transformer (PRT): Efficient Local Attention for Vision Transformers. 2026.
```

---

## 👨‍💻 Author

Bala Murugan K  
Independent Researcher  
Artificial Intelligence & Machine Learning  
