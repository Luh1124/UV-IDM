# UV-IDM: Identity-Conditioned Latent Diffusion Model for Face UV-Texture Generation

This repository contains the official code for the paper "[UV-IDM: Identity-Conditioned Latent Diffusion Model for Face UV-Texture Generation](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_UV-IDM_Identity-Conditioned_Latent_Diffusion_Model_for_Face_UV-Texture_Generation_CVPR_2024_paper.pdf)", presented at CVPR 2024.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
[![CVPR 2024](https://img.shields.io/badge/CVPR-2024-red.svg)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_UV-IDM_Identity-Conditioned_Latent_Diffusion_Model_for_Face_UV-Texture_Generation_CVPR_2024_paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-Coming_Soon-lightgrey.svg)](#)



## Abstract

3D face reconstruction aims to generate high-fidelity 3D face shapes and textures from single-view or multi-view images. However, current prevailing facial texture generation methods generally suffer from low-quality texture, identity information loss, and inadequate handling of occlusions. To solve these problems, we introduce an Identity-Conditioned Latent Diffusion Model for face UV-texture generation (UV-IDM) to generate photo-realistic textures based on the Basel Face Model (BFM). UV-IDM leverages the powerful texture generation capacity of a latent diffusion model (LDM) to obtain detailed facial textures. To preserve the identity during the reconstruction procedure, we design an identity-conditioned module that can utilize any in-the-wild image as a robust condition for the LDM to guide texture generation. UV-IDM can be easily adapted to different BFM-based methods as a high-fidelity texture generator. Furthermore, in light of the limited accessibility of most existing UV-texture datasets, we build a large-scale and publicly available UV-texture dataset based on BFM, termed BFM-UV. Extensive experiments show that our UV-IDM can generate high-fidelity textures in 3D face reconstruction within seconds while maintaining image consistency, bringing new state-of-the-art performance in facial texture generation.



## Next

- [ ] Update Gradio demo
- [ ] Release datasets
- [x] Release infer code


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:
* Python 3.8+ installed.
* Ensure you have GPU(s) with CUDA support (NVIDIA recommended).
* Install the necessary Python packages (listed in `requirements.txt`).

## Installation

To install this project, clone it using Git and install the dependencies:

```bash
git clone https://github.com/username/UV-IDM.git
cd UV-IDM
pip install -r requirements.txt
## Requirements
Please first download our checkpoint file in this link [google-link](https://drive.google.com/drive/folders/1ZgKWL_7aFnSUiCZTt6YVCT3oSxwBwTEn?usp=sharing), and place them in the folder.
- uv_idm_main/
    - checkpoints/
        - ...
    - pretrained/
        - ...
    - BFM/
        - ...
    - third_party/
        - ...
        A suitable environment named `ldm` can be created
        and activated with:

```
conda env create -f environment.yaml
conda activate ldm
git clone https://github.com/NVlabs/nvdiffrast
pip install -e nvdiffrast
```


## Test with UV-IDM
We recommend you to generate a filelist that contains the absolute path of your images.
A possible demo is in test_imgs.
Our network will generate three output, containing the render image, UV map and obj file.

## Quick start


You can start with our provided example by:
CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py --images_list_file test.txt --outdir test_imgs/output

## Start with your own imgs
CUDA_VISIBLE_DEVICES=0 python scripts/visualize.py --images_list_file your_txt_list --outdir your_output_path



