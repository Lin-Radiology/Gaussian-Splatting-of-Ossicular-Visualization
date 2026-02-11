# Gaussian Splatting–Based Three-Dimensional Visualization of the Ossicular Chain

This repository provides supplementary materials, evaluation scripts, and quantitative results supporting the manuscript:

**High-Resolution Three-Dimensional Visualization of the Ossicular Chain from Clinical CT Using Automated Segmentation and Gaussian Splatting**


---

## 1. Overview

This repository contains:

- Evaluation scripts for geometric consistency analysis
- Quantitative measurement outputs 
- Example Gaussian point cloud models (non-identifiable)
- Statistical summaries reported in the manuscript

This repository is provided to enhance transparency, reproducibility, and compliance with Springer Open data-sharing policies.

---

## 2. Data Availability Statement

Due to institutional and ethical restrictions, raw clinical CT datasets cannot be made publicly available.

The CT data were retrospectively collected from West China Hospital, Sichuan University, and are subject to institutional data protection regulations.

However, the following materials are publicly provided:

- Geometric evaluation scripts
- Surface-to-Gaussian distance measurements (.npy format)
- Aggregated quantitative metrics
- Representative Gaussian point cloud (.ply format, anonymized)

Researchers may request access to de-identified data subject to institutional approval.

---

## 3. Repository Structure

The repository is organized as follows:

```text
.
├── 0040025443_R_NBK.blend
├── Zoom_Resample_SigFile.py
├── calc_coverage.py                  # Script for surface coverage calculation
├── dataset.json                      # Dataset specification/config
├── geometry_consistency.py           # Surface-to-Gaussian distance computation
├── plans.json                        # Segmentation/nnUNet plans
├── point_cloud.ply                   # Example Gaussian splatting point cloud
├── postprocessing.json
├── postprocessing.pkl
├── progress.png                      # Training/evaluation progress visualization
├── summary.json                      # Summary results from evaluation/training
├── surface_to_gaussian_distances.npy # Precomputed geometry deviation arrays
├── training_log_2025_11_20_23_16_34.txt  # Training log for segmentation (if available)
├── README.md                         # This documentation
└── segmentation_surface_examples/    # Example reconstructed meshes (anonymized)
``` 
---

## 4. Reproducibility Instructions

This section outlines the overall workflow from automated segmentation to Gaussian splatting–based visualization and quantitative evaluation.

---

### Step 1. Automated Segmentation (nnU-Net)

Clinical CT images were converted to NIfTI format and organized following the nnU-Net v2 dataset structure.

Segmentation was performed using:

- **nnU-Net v2**  
  https://github.com/MIC-DKFZ/nnUNet  

Training and inference were conducted using the standard 3D full-resolution configuration. The output consisted of binary ossicular masks in `.nii.gz` format.

---

### Step 2.Resample and Zoom

The predicted segmentation masks were used to resample origin files and improve render efficiency.

Script:
```Zoom_Resample_SigFile.py```

Output:
- Surface mesh (`.ply`)

---

### Step 3. Blender-Based Visualization

Surface meshes were imported into:

- **Blender 4.2 LTS**
- Bioxel Nodes (v1.0.9)

Cinematic rendering was performed using Cycles, with multi-angle orbital camera placement to generate high-resolution visualization images.

---

### Step 4. Gaussian Splatting Optimization

Gaussian splatting models were generated using the official implementation:

https://github.com/graphdeco-inria/gaussian-splatting  

The optimized Gaussian representation was exported as:
```point_cloud/iteration_30000/point_cloud.ply```
Each case contained approximately 50,000–65,000 Gaussian primitives.

---

### Step 5. Geometric Evaluation

Geometric consistency between segmentation surfaces and Gaussian representations was computed using:
```geometry_consistency.py```
Surface coverage analysis was performed using:
```calc_coverage.py```

---

### Workflow Summary

Clinical CT  
→ nnU-Net segmentation  
→ Resample and Zoom
→ Blender visualization  
→ Gaussian splatting optimization  
→ Geometric consistency & coverage evaluation
---

## 5. Software Environment and External Dependencies

The visualization and segmentation pipeline was implemented using the following software and external frameworks:

### 3D Visualization and Rendering

- **Blender**: Version 4.2 LTS  
- **BlenderNeRF**: Blender add-on for neural rendering integration  
- **Bioxel Nodes**: Version 1.0.9  

### Gaussian Splatting Framework

- Official implementation of 3D Gaussian Splatting:  
  https://github.com/graphdeco-inria/gaussian-splatting  

The Gaussian optimization and point cloud generation were performed using the official implementation with default training parameters unless otherwise specified.

### Automated Segmentation

- **nnU-Net (v2)**:  
  https://github.com/MIC-DKFZ/nnUNet  

The segmentation network was trained on manually annotated ossicular masks using the standard nnU-Net configuration with region-of-interest cropping and intensity normalization.

---

## 6. Ethical Considerations

- All clinical data were retrospectively collected.
- Institutional review board approval was obtained.
- All datasets were anonymized before processing.
- No identifiable patient information is included in this repository.

---

## 7. Model Characteristics

Representative Gaussian models:

- Model size: 12–16 MB per case
- Number of Gaussian primitives: ~50,000–65,000
- Stored in `.ply` format

---

## 8. Intended Use

This repository is intended for:

- Methodological transparency
- Reproducibility verification
- Research and educational purposes

It is not intended for clinical diagnosis.

---


## 9. Contact

For academic inquiries:

Lin  
West China School of Medicine 
Sichuan University  
email:linhan@stu.scu.edu.cn
