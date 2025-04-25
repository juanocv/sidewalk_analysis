# Automatic Sidewalk Width Estimation and Obstacle Detection Using Panoptic Segmentation and Depth Estimation on Brazilian Street View Images

## Description
Have you ever thought about why sidewalks, specially in third-world countries, are so deficient
with little to no walkable space and plenty obstacles on the way?

We have thought about it too. 

That is the reason the main goal of this project is to use artificial intelligence tools, more specifically
computer vision models, to segment, detect and calculate the width of sidewalks and potential
obstacles that might be in a pedestrian's way.

This project explores and evaluates two advanced computer vision pipelines to automate sidewalk width estimation:
- Detectron2 + MiDaS (COCO dataset)
- OneFormer + MiDaS (ADE20k dataset)

The pipelines leverage panoptic segmentation models and depth estimation models to provide reliable and automated sidewalk measurement solutions.


## Workflow
TDB

## Technical Approach
### 1. Image Acquisition
- Images were acquired through Google Street View Static API.
- Standardized parameters were used, with default Field-of-View (FOV) and fixed intervals for compass heading angles.
### 2. Sidewalk Segmentation
- Two approaches were evaluated:
  - Detectron2: panoptic_fpn_R_101_dconv_cascade_gn_3x (COCO dataset)
  - OneFormer: shi-labs/oneformer_ade20k_swin_large (ADE20k dataset)
### 3. Depth Estimation
- Monocular depth estimation was performed using the MiDaS depth estimation model (DPT_Large).
### 4. Sidewalk Width Estimation
- Sidewalk width calculations were made using depth maps combined with segmentation masks. A custom methodology (pixel-to-meter conversion based on assumed camera parameters and MiDaS depth maps) was developed.
### 5. Obstacle Detection
- A separate functionality based on segmentation results was developed to identify potential obstacles on sidewalks.
## How does it work?
TDB

## Installation
First of all you have to clone this repo via ```git clone https://github.com/juanocv/sidewalk_analysis.git```

Secondly, you *must* install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), [OneFormer](https://github.com/SHI-Labs/OneFormer/blob/main/INSTALL.md) and their respective dependencies in the same folder you have cloned this repo.

Finally, you are good to go

>For how to install detectron2 on Windows please check [this guide](https://dev.to/reckon762/how-to-install-detectron2-on-windows-3hil)

## Examples
TDB

## Results
Detailed experimental evaluations comparing both pipelines are provided in the notebook and the scripts. Brief conclusions drawn from experiments include:
- OneFormer + MiDaS consistently produces lower relative error estimations (under 25%, sometimes even under 5%) compared to Detectron2.
- The optimal sidewalk width estimations depend significantly on the image acquisition angle, with diagonal views providing more accurate measurements.
- OneFormer demonstrates higher computational demand, approximately double the inference time compared to Detectron2.
- Challenges remain, especially when sidewalks visually resemble roads, highlighting the need for refined capture methods and additional depth cues.

## Important Notes and Future Work
- Current estimations assume a fixed camera Field-of-View (FOV); actual measurements could improve significantly with more precise calibration.
- Increasing the density of compass heading angles (currently at 30° intervals) could further enhance the accuracy and robustness of measurements.
- Future improvements include multi-view stereo approaches, camera calibration refinement, and integration with GIS databases for better spatial accuracy.

## Citation
If you use this repository in your research, please consider citing our work:
```
@misc{citation2025,
  author = {Diego Guerra, Juan Oliveira de Carvalho},
  title = {Automatic Sidewalk Width Estimation and Obstacle Detection Using Panoptic Segmentation and Depth Estimation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/juanocv/sidewalk-analysis},
}
```


