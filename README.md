# Comparison and Analysis of Vision Transformer Variants
Last year, ChatGPT successfully captured public attention, and now we are anticipating the arrival of the next generation. Our research analyzes two Transformer-based backbones, Vision Transformer and Swin Transformer, o identify the optimal architecture for computer vision tasks. We aim to demonstrate how Transformers can be applied to vision problems and to understand the reasoning behind the network’s decisions.

## Project Page
Please visit our project page! It documents our learning process, the challenges we encountered, and how we addressed them. We hope you enjoy exploring it.
[Comparison and Analysis of Vision Transformer Variants](https://sites.google.com/view/project666/home)

## System Environment  
   * python 3.9
   * Pytorch 2.0.1
   * CUDA 11.7

## Vision Transformer Variants
* Vision Transformer
  * original code: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer
  * early stopping code from: https://github.com/Bjarten/early-stopping-pytorch/tree/master
* Swin Transformer
  * microsoft: https://github.com/microsoft/Swin-Transformer
  * original code: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/swin_transformer

 
## Visualization tools
* GradCam
   * original code: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam
* SorceCam
   * original code: https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md

## Visualization Results

|               | **Swin**        | **ViT**         |
|---------------|----------------|-----------------|
| **Grad-CAM**  | ![](assets/swin_616_gradcam.jpg) | ![](assets/vit_616_None_gradcam_cam.jpg) |
| **Score-CAM** | ![](assets/swin_616_scorecam.jpg) | ![](assets/vit_616_None_scorecam_cam.jpg) |
