# 2D Gum-Net

### Unsupervised Dense Deformation Grid Prediction for Elastic Fingerprint Alignment

<p align=\"center\">
  <a href=\"https://cvpr.thecvf.com/\"><img src=\"https://img.shields.io/badge/CVPR-2026-blue?style=for-the-badge\" alt=\"CVPR 2026\"></a>
  <a href=\"#\"><img src=\"https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white\" alt=\"PyTorch\"></a>
  <a href=\"#license\"><img src=\"https://img.shields.io/badge/License-MIT-green?style=for-the-badge\" alt=\"License\"></a>
  <a href=\"#\"><img src=\"https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white\" alt=\"Python\"></a>
</p>

<p align=\"center\">
  <a href=\"#-overview\">Overview</a> •
  <a href=\"#-key-contributions\">Contributions</a> •
  <a href=\"#-architecture\">Architecture</a> •
  <a href=\"#-results\">Results</a> •
  <a href=\"#-quick-start\">Quick Start</a> •
  <a href=\"#-citation\">Citation</a>
</p>

<br>

**TL;DR:** We introduce an unsupervised deep learning framework that predicts dense deformation fields for elastic fingerprint alignment—eliminating the need for ground-truth labels while significantly reducing False Rejection Rates.

<br>

<img src=\"assets/gumnet_pipeline.png\" alt=\"2D Gum-Net Pipeline\" width=\"95%\"/>

<sub><b>Figure 1.</b> The 2D Gum-Net pipeline: Given a template S<sub>a</sub> and distorted impression S<sub>b</sub>, our framework extracts features via DCT spectral pooling, computes bidirectional correlation maps, and predicts a dense deformation field for pixel-wise alignment—all without supervision.</sub>

</div>

<br>

---

## 🎯 Overview

> *\"The fundamental challenge in fingerprint recognition isn't matching—it's alignment.\"*

Elastic skin deformation during capture introduces **non-linear distortions** that cause genuine fingerprint pairs to appear different. Traditional approaches either:
- Require expensive ground-truth deformation labels (impractical to obtain)
- Rely on minutiae extraction (fails under severe distortion)
- Use rigid transformations (insufficient for elastic deformation)

**2D Gum-Net** solves this by learning to predict dense deformation fields in a completely **unsupervised** manner, adapting techniques from structural biology (cryo-electron tomography) to the biometrics domain.

<br>

## ✨ Key Contributions

<table>
<tr>
<td width=\"50%\" valign=\"top\">

### 1️⃣ Cross-Domain Adaptation
First successful adaptation of 3D subtomogram alignment networks to 2D fingerprint biometrics, demonstrating that unsupervised structural alignment principles generalize across vastly different domains.

### 2️⃣ Unsupervised DDF Prediction
Novel Dense Spatial Transformer (DST) module enables end-to-end learning of pixel-wise deformation fields without any ground-truth supervision—solving the label acquisition bottleneck.

</td>
<td width=\"50%\" valign=\"top\">

### 3️⃣ DCT Spectral Pooling
Frequency-domain pooling layers that preserve ridge and valley microstructure significantly better than conventional spatial pooling, critical for maintaining discriminative fingerprint features.

### 4️⃣ Robust Pre-Match Rectification  
Effective mitigation of elastic torque and non-linear stretching, enabling substantial reduction in False Rejection Rate (FRR) even under severe distortion conditions.

</td>
</tr>
</table>

<br>

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           2D Gum-Net Framework                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐      ┌──────────────────┐      ┌───────────────────────┐    │
│   │ Template │─────▶│                  │      │                       │    │
│   │   Sₐ     │      │     Feature      │─────▶│   Siamese Matching    │    │
│   └──────────┘      │    Extraction    │      │                       │    │
│                     │   + DCT Pooling  │      │  • Correlation Maps   │    │
│   ┌──────────┐      │                  │─────▶│  • Cₐᵦ, Cᵦₐ          │    │
│   │Impression│─────▶│                  │      │                       │    │
│   │   Sᵦ     │      └──────────────────┘      └───────────┬───────────┘    │
│   └──────────┘                                            │                │
│                                                           ▼                │
│                                              ┌───────────────────────┐     │
│   ┌──────────┐      ┌──────────────────┐     │  Dense Spatial        │     │
│   │ Aligned  │◀─────│  Deformation     │◀────│  Transformer (DST)    │     │
│   │   Sᵦ'    │      │  Field Φ        │     │                       │     │
│   └──────────┘      └──────────────────┘     │  • Control Points     │     │
│                                              │  • Bicubic Interp.    │     │
│                                              └───────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Details

| Module | Components | Output |
|:-------|:-----------|:-------|
| **Feature Extraction** | CNN backbone + DCT spectral pooling layers | Feature maps v<sub>a</sub>, v<sub>b</sub> |
| **Siamese Matching** | Symmetric 2D correlation layer | Bidirectional correlation maps C<sub>ab</sub>, C<sub>ba</sub> |
| **Non-linear Alignment** | Control point regression + Bicubic interpolation | Dense deformation field Φ |
| **Spatial Transformer** | Differentiable grid sampling | Aligned impression S<sub>b</sub>' |

<br>

### 🔬 DCT Spectral Pooling: Why It Matters

<div align=\"center\">
<img src=\"assets/dct_viz.png\" alt=\"DCT Spectral Pooling Comparison\" width=\"90%\"/>

<sub><b>Figure 2.</b> Comparison of pooling methods across subsampling factors (1:1 → 16:16). DCT spectral pooling preserves ridge structure and fine details significantly better than max or average pooling at aggressive downsampling rates—critical for maintaining discriminative fingerprint features.</sub>
</div>

<br>

**The Problem:** Standard pooling operations (max, average) destroy high-frequency information essential for fingerprint matching.

**Our Solution:** DCT spectral pooling operates in the frequency domain, selectively retaining the most informative coefficients:

```python
# Conceptual DCT Spectral Pooling
def dct_spectral_pool(x, output_size):
    # Transform to frequency domain
    X_dct = dct_2d(x)
    
    # Crop to retain low-frequency components
    X_cropped = X_dct[:, :, :output_size[0], :output_size[1]]
    
    # Transform back to spatial domain
    return idct_2d(X_cropped)
```

<br>

### Training Objective

The network optimizes an unsupervised objective combining structural alignment and spatial smoothness:

$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{dice}} + \lambda \cdot \mathcal{L}_{	ext{smooth}}$$

| Loss Component | Description | Purpose |
|:---------------|:------------|:--------|
| $\mathcal{L}_{	ext{dice}}$ | Soft Dice coefficient between aligned S<sub>b</sub>' and template S<sub>a</sub> | Maximize structural overlap |
| $\mathcal{L}_{	ext{smooth}}$ | Spatial gradient penalty on deformation field Φ | Enforce physically plausible deformations |

<br>

---

## 📊 Results

### Alignment Performance

Our method demonstrates robust alignment across diverse conditions:

<table>
<tr>
<td align=\"center\"><b>✓ High Distortion Tolerance</b><br><sub>Maintains accuracy under severe elastic deformation</sub></td>
<td align=\"center\"><b>✓ Noise Robustness</b><br><sub>Consistent across 5 SNR levels</sub></td>
<td align=\"center\"><b>✓ Universal Generalization</b><br><sub>Effective on all 7 Henry classification types</sub></td>
</tr>
</table>

### Qualitative Results

<div align=\"center\">

| Template S<sub>a</sub> | Distorted S<sub>b</sub> | Aligned S<sub>b</sub>' | Deformation Field Φ |
|:----------------------:|:-----------------------:|:----------------------:|:-------------------:|
| <img src=\"assets/results/template.png\" width=\"120\"/> | <img src=\"assets/results/distorted.png\" width=\"120\"/> | <img src=\"assets/results/aligned.png\" width=\"120\"/> | <img src=\"assets/results/deformation.png\" width=\"120\"/> |

<sub>Example alignment on a never-before-seen fingerprint pair. The predicted deformation field successfully rectifies non-linear elastic distortion.</sub>

</div>

<br>

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/anonymous/2d-gumnet.git
cd 2d-gumnet

# Create environment
conda create -n gumnet python=3.9 -y
conda activate gumnet

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Inference

```python
import torch
from gumnet import GumNet2D, load_image, visualize_alignment

# Load pretrained model
model = GumNet2D.from_pretrained(\"gumnet-base\")
model.eval()

# Load fingerprint pair
template = load_image(\"examples/template.png\")      # Reference fingerprint
impression = load_image(\"examples/impression.png\")  # Distorted impression

# Predict alignment
with torch.no_grad():
    aligned, deformation_field = model(template, impression)

# Visualize results
visualize_alignment(template, impression, aligned, deformation_field, 
                    save_path=\"output/alignment_result.png\")
```

### Training

```bash
# Train on synthetic dataset
python train.py \
    --config configs/synthetic.yaml \
    --data_path data/anguli \
    --output_dir checkpoints/

# Resume training
python train.py \
    --config configs/synthetic.yaml \
    --resume checkpoints/latest.pth
```

### Evaluation

```bash
# Evaluate alignment quality
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_path data/test \
    --output_dir results/
```

<br>

---

## 📁 Repository Structure

```
2d-gumnet/
├── configs/                  # Training configurations
│   ├── synthetic.yaml
│   └── custom.yaml
├── gumnet/                   # Core library
│   ├── __init__.py
│   ├── model.py              # GumNet2D architecture
│   ├── modules/
│   │   ├── feature_extractor.py
│   │   ├── siamese_matching.py
│   │   ├── spatial_transformer.py
│   │   └── dct_pooling.py    # DCT spectral pooling
│   ├── losses.py             # Loss functions
│   └── utils.py              # Utilities
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── visualize.py          # Visualization tools
├── data/                     # Dataset directory
├── checkpoints/              # Model weights
├── assets/                   # README assets
├── requirements.txt
└── README.md
```

<br>

---

## 📦 Dataset

We utilize synthetic fingerprints generated with the **Anguli** fingerprint generator:

| Attribute | Details |
|:----------|:--------|
| **Total Images** | 73,500 |
| **SNR Levels** | 5 (varying noise conditions) |
| **Classification Types** | 7 (Henry Classification) |
| **Types** | Arch, Tented Arch, Right Loop, Left Loop, Whorl, Twin Loop, Accidental |

<br>

---

## 📅 Release Timeline

| Asset | Status | Expected |
|:------|:------:|:---------|
| 📄 Paper | ✅ Submitted | CVPR 2026 |
| 💻 Training Code | ⏳ Pending | Upon Acceptance |
| 💻 Evaluation Code | ⏳ Pending | Upon Acceptance |
| 🔧 Pre-trained Weights | ⏳ Pending | Camera-Ready |
| 📊 Synthetic Dataset | ⏳ Pending | Upon Acceptance |
| 🎮 Demo | ⏳ Pending | Camera-Ready |

<br>

---

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{gumnet2d2026,
    title     = {2D Gum-Net: Unsupervised Dense Deformation Grid Prediction 
                 for Elastic Fingerprint Alignment},
    author    = {Anonymous},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision 
                 and Pattern Recognition (CVPR)},
    year      = {2026}
}
```

<br>

---

## Acknowledgments

This work builds upon the foundational Gum-Net architecture developed for cryo-electron tomography subtomogram alignment. We thank the original authors for their pioneering contributions to unsupervised structural alignment.

<br>

---

## License

This project is released under the [MIT License](LICENSE).

<br>

---

<div align=\"center\">

**Anonymous CVPR 2026 Submission**

<br>

<sub>If you have any questions, please open an issue or contact us after the review process.</sub>

<br>

⭐ **Star this repository if you find it helpful!** ⭐

</div>
"
## Important Links:

- [ICML New In ML Affinity Event](https://newinml.github.io/NewInML2025ICML/)
- [Gum-Net Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Gum-Net_Unsupervised_Geometric_Matching_for_Fast_and_Accurate_3D_Subtomogram_CVPR_2020_paper.pdf)
- [Gum-Net Supplemental](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Zeng_Gum-Net_Unsupervised_Geometric_CVPR_2020_supplemental.pdf#page=3.00)
- [Google Drive](https://drive.google.com/drive/folders/1iDYChJ-Ee0wIsDdZsaEcUtGYgJdXz5y2)
- [Current Gold Standard Fingerprint Matching Methods](https://docs.google.com/spreadsheets/d/1cSLaRhZ0j-iSGLZHZiXBqevQrt3hOLhBJd39syqC32s/edit?usp=sharing)
- [Dates and Deadlines](https://icml.cc/Conferences/2026/Dates) 
- [A Curated List of Fingerprint Datasets.](https://github.com/robertvazan/fingerprint-datasets)
