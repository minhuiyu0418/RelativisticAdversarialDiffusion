# Relativistic Adversarial Diffusion (RelA-Diffusion)

This repository contains the official implementation of [**RelA-Diffusion: Relativistic Adversarial Diffusion for Multi-Tracer PET Synthesis from Multi-Sequence MRI**](https://arxiv.org/abs/2602.21345).

<img width="4168" height="1709" alt="Image" src="https://github.com/user-attachments/assets/ab4c0fb2-4d9b-4ee1-88f5-2f3a8ea01bb0" />

---

## 🛠️ Installation & Environment

### 1. Conda Setup 
```bash
conda env create -f environment.yml
conda activate pytorch_3

```

### 2. Pip Setup 

```bash
pip install -r requirements.txt

```

---
## 📂 Repository Structure

* `RelADiff/`: Standalone directory of a python project containing model architectures, trainers, and the `run.py` entry point.
* `generative/`: Module dedicated to the generative components of the diffusion process.
* `checkpoints/`: Directory for pre-trained diffusion model weights.
* `discriminator_checkpoints/`: Directory for adversarial discriminator weights.

---

## 📊 Data Preparation

The model expects preprocessed 3D NIfTI brain images. Images should be co-registered to MNI space. A common pro-processing pipeline can be found [here](https://www.gaaindata.org/data/centiloid/Centiloid_Processing.docx).

Organize your data as follows:

```text
data_root/
  ├── 00001/             # subject ID
  │   ├── 00001_T1.nii   # Input Condition 1
  │   ├── 00001_T2.nii   # Input Condition 2
  │   ├── 00001_PBR.nii  # Target Ground Truth PBR
  │   ├── 00001_PIB.nii  # Target Ground Truth PIB
  │   └── 00001_TAU.nii  # Target Ground Truth TAU
  └── 00002/ ...
```

### 💡 Implementation Details

The default data loader in `RelADiff/data.py` is configured for **Multi-Tracer Consistency**. It filters for subjects with a complete set of modalities (T1, T2, PBR, PIB, TAU) to ensure a high-quality, balanced validation set.

#### **Using Incomplete Modalities?**

If your dataset does not have all five modalities for every subject, the validation pool will be empty. To bypass this:

* **Option A**: Reduce the `val_size` parameter in the config.
* **Option B**: Switch to a global subject pool for validation. We have provided a pre-commented Option B in `data.py`.
---

## 🚀 Quick Start (Running the Project)

You can run the entire pipeline (Preprocessing, Training, and Inference) using the convenience runner.

From the project root:

```bash
python run.py --mode all
```

*Alternatively, for interactive experimentation and visualization, refer to the [MRI_to_MultiTracer_PET_Synthesizing.ipynb](https://www.google.com/search?q=./MRI_to_MultiTracer_PET_Synthesizing.ipynb) notebook.*


The [TorchIO](https://www.google.com/search?q=https://torchio.readthedocs.io/) pipeline automatically handles cropping to $(160, 180, 160)$ and intensity normalization to $[-1, 1]$.

---

## 📦 Pre-trained Models

Download the weights for different tracers and place them in the `checkpoints/` folder:

* [Download Pretrained Weights](https://github.com/minhuiyu0418/RelativisticAdversarialDiffusion/blob/main/checkpoints/epoch99_checkpoint.pt)

---

## ✍️ Citation

If you use this code or our paper in your research, please cite:

```bibtex
@article{yu2026reladiffusion,
  title={RelA-Diffusion: Relativistic Adversarial Diffusion for Multi-Tracer PET Synthesis from Multi-Sequence MRI},
  author={Yu, Minhui and Sun, Yongheng and Lalush, David S. and Mihalik, Jason P. and Yap, Pew-Thian and Liu, Mingxia},
  journal={arXiv preprint arXiv:2602.21345},
  year={2026}
}

```
---
## 📜 License

This project is licensed under the Apache-2.0 License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

