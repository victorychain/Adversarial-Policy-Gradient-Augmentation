# [Adversarial Policy Gradient for Medical Deep Learning Image Augmentation (MICCAI2019)](https://arxiv.org/abs/1909.04108)

Kaiyang Cheng *, Claudia Iriondo *, Francesco Calivá,  Justin Krogue, Sharmila Majumdar, Valentina Pedoia

*Equal contribution

## Introduction

The use of semantic segmentation for masking and cropping input images has proven to be a significant aid in medical imaging classification tasks by decreasing the noise and variance of the training dataset. However, implementing this approach with classical methods is challenging: the cost of obtaining a dense segmentation is high, and the precise input area that is most crucial to the classification task is difficult to determine a-priori. We propose a novel joint-training deep reinforcement learning framework for image augmentation. A segmentation network, weakly supervised with policy gradient optimization, acts as an agent, and outputs masks as actions given samples as states, with the goal of maximizing reward signals from the classification network. In this way, the segmentation network learns to mask unimportant imaging features. Our method, Adversarial Policy Gradient Augmentation (APGA), shows promising results on Stanford's MURA dataset and on a hip fracture classification task with an increase in global accuracy of up to 7.33% and improved performance over baseline methods in 9/10 tasks evaluated. We discuss the broad applicability of our joint training strategy to a variety of medical imaging tasks.

![image](images/FINALFINAL.png)

## Usage

1. Requirements:

   - Python 3.6

   - PyTorch 1.0+

   - Torchvision

   - Numpy

   - Pandas

   - Tqdm

   - [Google Fire](https://github.com/google/python-fire)

     

2. Clone the repository:

   ```shell
   git clone https://github.com/victorychain/Adversarial-Policy-Gradient-Augmentation.git
   ```

3. Dataset

   - Download the [MURA](https://stanfordmlgroup.github.io/competitions/mura/) dataset
   - Please put dataset in folder `./datasets`

4. Training:

   To train all the methods by body parts (elbow for example):

   ```shell
   python apga_mura.py --seed 88 --body-part elbow --n-runs 1 --gpu-id 0 --train-cutout 1 --train-apga 1 --train-gradcam 1 --train-end2end 1
   ```

   To train k-shot:

   ```shell
   python apga_mura.py --seed 88 --body-part elbow --n-runs 1 --gpu-id 0 --train-cutout 1 --train-apga 1 --train-gradcam 1 --train-end2end 1 --n-shot 100
   ```

   

## Citation

If APGA is useful for your research, please consider citing:

```
 @article{Cheng_Iriondo_Calivá_Krogue_Majumdar_Pedoia_2019, 
 title={Adversarial Policy Gradient for Deep Learning Image Augmentation}, 
 url={http://arxiv.org/abs/1909.04108}, 
 note={arXiv: 1909.04108}, 
 journal={arXiv:1909.04108 [cs]}, 
 author={Cheng, Kaiyang and Iriondo, Claudia and Calivá, Francesco and Krogue, Justin and Majumdar, Sharmila and Pedoia, Valentina}, 
 year={2019}, 
 month={Sep} }

```

## Acknowledgement

Thank you UCSF and UC Berkeley!