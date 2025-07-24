# Dynamic Multi-Image Weighting for Automated Detection and Diagnosis of Abnormal Urinary Tract on VCUG

This repository provides the official implementation of our paper:

> **Dynamic Multi-Image Weighting for Automated Detection and Diagnosis of Abnormal Urinary Tract on Voiding Cystourethrography with a Deep Learning System: A Retrospective, Large-Scale, Multicenter Study**  
> Min Wu, Zhanchi Li, Yidong Liu, Zelong Tan, Wenjuan Tang, Xiaoqi Xuan, Hui Feng, Weihua Cao, Ning Ding, Bojun Wang, Zheyuan Wang, Likai Zhuang, 

> Published in *Research*, AAAS.
📄 [**Paper Link**](https://spj.science.org/doi/10.34133/research.0771)  


---

## 🔍 Overview

We present **VCUG-DAM**, an artificial intelligence system designed for **automatic segmentation and diagnosis** of the bladder, urethra, and ureters using **multiple VCUG images**. The model dynamically weights the importance of each image during inference and supports multitask learning.

### Key Features
- Multi-task deep learning system for VCUG images.
- Dynamic image importance weighting.
- Accurate diagnosis of bladder, urethral, and vesicoureteral reflux (VUR) conditions.
- Evaluation on **7,899 VCUG images** from **1,660 patients** across **15 Chinese hospitals**.
- Improves clinicians' diagnostic performance significantly.

---

## 📈 Performance Highlights

| Task         | AUC (Model Only) | AUC (With AI Assistance) |
|--------------|------------------|---------------------------|
| **Bladder**  | 0.8772           | 0.9456                    |
| **Urethra**  | 0.7752           | 0.7943                    |
| **Left VUR** | 0.9443           | 0.9641                    |
| **Right VUR**| 0.9342           | 0.9506                    |

> All improvements in AUC were statistically significant (*P* < 0.0001).

---


## 🏗️ Project Structure

This repo includes code for multi-task **segmentation** and **classification** of VCUG images.

Please organize your data as follows:

```
data/
├── images/
│   ├── image1.png
│   ├── image2.png
├── masks/
│   ├── mask1.png
│   ├── mask2.png
└── labels.csv
```

---

## 🧠 Pretrained Model

Please download the pretrained MAE checkpoint from:

🔗 [MAE ViT-Base Pretrained Weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)

---

## 🚀 Quick Start

Run the following command to fine-tune the model:

```bash
python main.py \
  --input_size 512 \
  --batch_size 16 \
  --data_path path/to/yourdata \
  --finetune path/to/your/checkpoint
```

**Example:**

```bash
python main.py \
  --input_size 512 \
  --batch_size 16 \
  --data_path ./vcug2/data \
  --finetune ./vcug2/checkpoints/mae_finetuned_vit_base.pth
```

---

## 🏥 Study Design

1. **Data Collection**  
   - From 15 hospitals (Center A: Children’s Hospital of Fudan University; Center B: 10 hospitals; Center C: 4 hospitals)
2. **Model Development**  
   - Segmentation + Classification at image and patient level
3. **Clinician Evaluation**  
   - 12 clinicians with and without AI assistance
4. **Model Analysis**  
   - Performance, attention, consistency

---



📌 BibTeX citation:
```bibtex
@article{wudynamic,
  title={Dynamic Multi-Image Weighting for Automated detection and diagnosis of abnormal urinary tract on Voiding Cystourethrography with a deep learning system: a retrospective, large-scale, multicentrer study},
  author={Wu, Min and Li, Zhanchi and Liu, YiDong and Tan, Zelong and Tang, Wenjuan and Xuan, Xiaoqi and Feng, Hui and Cao, Weihua and Ding, Ning and Wang, BoJun and others},
  journal={Research},
  publisher={AAAS}
}
```