# Gaussian Splatting–Based Three-Dimensional Visualization of the Ossicular Chain

This repository provides supplementary materials, evaluation scripts, and quantitative results supporting the manuscript:

**High-Resolution Three-Dimensional Visualization of the Ossicular Chain from Clinical CT Using Automated Segmentation and Gaussian Splatting**

Submitted to *Insights into Imaging* (Springer Open).

---

## 1. Overview

This repository contains:

- Evaluation scripts for geometric consistency analysis
- Quantitative measurement outputs (machine-readable format)
- Example Gaussian point cloud models (non-identifiable)
- Instructions for reproducing geometric metrics
- Statistical summaries reported in the manuscript

This repository is provided to enhance transparency, reproducibility, and compliance with Springer Open data-sharing policies.

---

## 2. Data Availability Statement

Due to institutional and ethical restrictions, raw clinical CT datasets cannot be made publicly available.

The CT data were retrospectively collected from West China Hospital, Sichuan University, and are subject to institutional data protection regulations.

However, the following materials are publicly provided:

- Geometric evaluation scripts
- Surface-to-Gaussian distance measurements (.npy format)
- Summary statistics (.csv format)
- Aggregated quantitative metrics
- Representative Gaussian point clouds (.ply format, anonymized)

Researchers may request access to de-identified data subject to institutional approval.

---

## 3. Repository Structure
geometry_evaluation/
├── geometry_consistency.py
├── coverage_evaluation.py
├── example_distances/
│   ├── case1_surface_to_gaussian.npy
│   ├── case2_surface_to_gaussian.npy
│   └── case3_surface_to_gaussian.npy
├── summary_metrics.csv
├── combined_cdf_figure.png
└── segmentation_surface_examples/
---

## 4. Quantitative Metrics Reported

The following quantitative metrics were computed and reported in Section 3.2 of the manuscript:

### Geometric Consistency (Surface-to-Gaussian Distance)

| Metric | Mean ± SD (mm) |
|--------|----------------|
| Mean distance | 0.132 ± 0.037 |
| Median distance | 0.115 ± 0.032 |
| 90th percentile | 0.237 ± 0.073 |

### Surface Coverage (Threshold = 0.3 mm)

| Case | Coverage (%) |
|------|--------------|
| Case 1 | 90.75% |
| Case 2 | 99.48% |
| Case 3 | 92.00% |
| **Overall** | **94.08% ± 3.86%** |

All data are provided in machine-readable `.csv` format.

---

## 5. Reproducibility Instructions

### Requirements

- Python ≥ 3.9
- nibabel
- trimesh
- open3d
- numpy
- scipy
- matplotlib

### Step 1: Generate Surface Mesh from Segmentation

Use `geometry_consistency.py` to:

- Convert NIfTI mask to surface mesh (Marching Cubes)
- Load Gaussian point cloud
- Compute surface-to-Gaussian nearest-neighbor distances

### Step 2: Compute Coverage Ratio

Use `coverage_evaluation.py` to calculate:

Coverage = proportion of surface points within 0.3 mm of Gaussian representation

### Step 3: Generate Combined Visualization

The script produces:

- Histogram of geometric distances
- Cumulative distribution function (CDF)
- Combined publication-ready figure

---

## 6. Machine-Readable Files

To comply with Springer Open policy, all quantitative data are provided in:

- `.csv` (tabular summary)
- `.npy` (raw distance arrays)
- `.ply` (3D point cloud format)

No data are embedded in PDF-only format.

---

## 7. Ethical Considerations

- All clinical data were retrospectively collected.
- Institutional review board approval was obtained.
- All datasets were anonymized before processing.
- No identifiable patient information is included in this repository.

---

## 8. Model Characteristics

Representative Gaussian models:

- Model size: 12–16 MB per case
- Number of Gaussian primitives: ~50,000–65,000
- Stored in `.ply` format

---

## 9. Intended Use

This repository is intended for:

- Methodological transparency
- Reproducibility verification
- Research and educational purposes

It is not intended for clinical diagnosis.

---

## 10. Citation

If you use this work, please cite:

[Manuscript citation information will be added upon acceptance.]

---

## 11. Contact

For academic inquiries:

Lin  
Department of Radiology  
West China Hospital  
Sichuan University  
