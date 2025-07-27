
# Few-Shot Deployment of Pretrained MRI Transformers

This repository contains the official implementation of the paper:

**Few-Shot Deployment of Pretrained MRI Transformers in Brain Imaging Tasks**  
Mengyu Li, Guoyao Shen, Chad W. Farris, Xin Zhang  
Department of Mechanical Engineering & The Photonics Center, Boston University  
Boston Medical Center and Boston University School of Medicine  
*Corresponding author: xinz@bu.edu*

---

## 🧠 Overview

This project presents a scalable framework for deploying **pretrained MAE-based vision transformers** in **few-shot brain MRI tasks**, including:

- **MRI Sequence Classification**
- **Skull Stripping**
- **Multi-class Anatomical Segmentation**

We introduce:
- **MAE-classify**: A lightweight classifier using a frozen MAE encoder.
- **MAE-FUnet**: A hybrid CNN-transformer segmentation model combining U-Net and MAE latent features via multiple fusion strategies.

Pretraining was performed on over **31 million brain MRI slices** from NACC, ADNI, OASIS, RadImageNet, and FastMRI.

---

## 🗂️ Directory Structure

```
.
├── demo_data/                     # Sample data for demo purposes
├── modeling/                     # Core model construction and wrappers
├── multi_seg/                   # Multi-class segmentation routines
├── nets/                        # Model architecture definitions (MAE, UNet, etc.)
├── sequence_detection/         # MRI sequence classification logic
├── tests/                       # Unit tests
├── train/                       # Training pipeline (data loader, trainer, logging)
├── utils/                       # Utility functions
│
├── help_func.py                 # Helper utilities for training/metrics
├── multi_segmentation_demo.ipynb         # Run multi-class segmentation (inference only)
├── multi_segmentation_finetune.ipynb     # Finetune segmentation model on few-shot data
├── sequence_detection_demo.ipynb         # Run classification (inference only)
├── sequence_detection_finetune.ipynb     # Finetune classifier on few-shot samples
├── skull_strip_demo.ipynb                # Skull stripping inference
├── skull_strip_finetune.ipynb            # Skull stripping fine-tuning
│
├── LICENSE
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/mri-transformer-fewshot.git
cd mri-transformer-fewshot

# (Optional) create conda environment
conda create -n mri-fewshot python=3.10
conda activate mri-fewshot

# Install dependencies
pip install -r requirements.txt
```

> Make sure you have PyTorch >= 1.13, torchvision, numpy, nibabel, and scikit-learn.

---

## 🧪 Usage & Demos

### 🔍 1. MRI Sequence Classification

**Demo:**  
```bash
Run `sequence_detection_demo.ipynb`
```

**Few-shot fine-tuning:**  
```bash
Run `sequence_detection_finetune.ipynb`
```

This loads a frozen MAE encoder and a lightweight classifier head for predicting MRI sequences (T1, T2, FLAIR, etc.).

---

### 🧠 2. Skull Stripping

**Demo:**  
```bash
Run `skull_strip_demo.ipynb`
```

**Few-shot fine-tuning:**  
```bash
Run `skull_strip_finetune.ipynb`
```

Performs binary brain extraction from T1, T2, PD, FLAIR, or DWI images.

---

### 🧩 3. Multi-Class Anatomical Segmentation

**Demo:**  
```bash
Run `multi_segmentation_demo.ipynb`
```

**Few-shot fine-tuning:**  
```bash
Run `multi_segmentation_finetune.ipynb`
```

Segments up to 13 anatomical structures from T1-weighted brain MRIs. Results are evaluated using Dice and IoU scores.

---

## 🧾 Pretrained Weights

Pretrained MAE weights trained on 31M slices are available on [Google Drive / HuggingFace 🤗 link TBD].

Download and place them in:

```
./nets/checkpoints/mae_pretrained/
```

---

## 📊 Reproducing Results

All experiments reported in the paper can be reproduced using the fine-tune notebooks. Adjust `stride`, `sample_size`, and `dataset` parameters to match each benchmark:

- **Sequence Detection** (Section 3.1)
- **Skull Stripping (NFBS, SynthStrip)** (Section 3.2)
- **Multi-class Segmentation (MRBrainS18, NACC)** (Section 3.3)
- **Ablation Studies** (Section 4)

---

## 📚 Citation

If you find this repository helpful, please cite our work:

```
@article{li2025fewshotMRI,
  title={Few-Shot Deployment of Pretrained MRI Transformers in Brain Imaging Tasks},
  author={Li, Mengyu and Shen, Guoyao and Farris, Chad W. and Zhang, Xin},
  journal={Scientific Reports},
  year={2025}
}
```

---

## 🧠 Keywords

- Few-shot learning  
- Brain MRI segmentation  
- Vision transformers  
- Masked Autoencoders (MAE)  
- Hybrid CNN-transformer models  
- Medical image classification

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📬 Contact

For questions or collaborations, please contact:  
**Prof. Xin Zhang** – *xinz@bu.edu*  
Boston University Photonics Center
