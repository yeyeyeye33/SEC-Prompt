# SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  


Official repository for our CVPR 2025 paper:  
**“SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning”**

---

## 📖 Table of Contents

- [About](#about)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Dataset Preparation & Usage](#dataset-preparation--usage)  
  - [Place data under `./data`](#place-data-under-datadata)  
  - [Download datasets](#download-datasets)  
  - [Run experiments](#run-experiments)  
- [Data Selection](#data-selection)   
- [Acknowledgements & License](#acknowledgements--license)  

---

## About

This repository contains the official implementation of **SEC-Prompt**, our CVPR 2025 submission.  
Due to work commitments, the code has not been exhaustively verified.  
If you encounter any bugs or discrepancies, please [open an issue](https://github.com/yeyeyeye33/SEC-Prompt/issues). 
And we plan to release the model weights corresponding to our experimental results in the future.

---

## Requirements


- See `requirements.txt` for the full list of dependencies and exact versions.

> ⚠️ **Important**: Install dependencies using the **exact versions** in `requirements.txt`.  
> Mismatched versions may lead to significantly different results (higher or lower), as observed in our experiments.

---

## Installation

```bash
git clone https://github.com/yeyeyeye33/SEC-Prompt.git
cd SEC-Prompt
pip install -r requirements.txt
```

---

## Dataset Preparation & Usage

### Place data under `./data`

Your project directory should look like:
```text
SEC-Prompt/
├── data/
│   ├── cifar100/           # auto-downloaded
│   ├── cub200/
│   └── imagenet_r/
├── data/index_list/
│   └── README.md
└── ...
```

### Download datasets

- **CIFAR-100**: automatically downloaded by the code.  
- **CUB-200**: [Google Drive](https://drive.google.com/file/d/1jx0ICqvgaXyfWUVLTv6St0_b6p7D0Hpm/view?usp=sharing)  
- **ImageNet-R**: [Google Drive](https://drive.google.com/file/d/1R4bRjYXnbRWje6hw_YPdsKr1HuojXzqO/view?usp=sharing)

### Run experiments

```bash
# CIFAR-100
python main.py --config ./exps/cifar.json

# CUB-200
python main.py --config ./exps/cub.json

# ImageNet-R
python main.py --config ./exps/inr.json
```

---

## Data Selection

For class splits and index lists, see:  
`./data/index_list/README.md`



## Acknowledgements & License

This work builds upon these excellent repos:

- [FSCIL-ASP (DawnLIU35)](https://github.com/DawnLIU35/FSCIL-ASP)  
- [CODA-Prompt (GT-RIPL)](https://github.com/GT-RIPL/CODA-Prompt)  

We sincerely thank the authors for sharing their code.

## Citation

If you find our work useful, please cite our CVPR 2025 paper:  

```bibtex
@inproceedings{liu2025sec,
  title={SEC-Prompt: SEmantic Complementary Prompting for Few-Shot Class-Incremental Learning},
  author={Liu, Ye and Yang, Meng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25643--25656},
  year={2025}
