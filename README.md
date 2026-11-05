# SR-Weather

**A super-resolution framework for downscaling machine learning weather prediction toward 1-km air temperature**

[![Paper](https://img.shields.io/badge/Paper-npj%20Climate%20and%20Atmospheric%20Science-blue)](https://doi.org/10.1038/s41612-026-01328-5)
![Python](https://img.shields.io/badge/Python-3.9.12-blue)

## Abstract

> Artificial intelligence has significantly advanced global weather forecasting, yet the coarse spatial resolution of existing prediction systems limits their ability to capture fine-scale temperature variations associated with complex terrain and urban environments. In this study, we propose **SR-Weather**, a deep learning-based super-resolution framework that downscales 0.25° weather forecasts to 1-km air temperature fields using MODIS-derived air temperature and high-resolution auxiliary data. The framework incorporates topography, impervious surface fraction, and seasonal climatology maps (SCM) to represent spatial temperature heterogeneity. When applied to FuXi forecasts over South Korea, SR-Weather reduces the 7-day forecast error by more than 20% and effectively reconstructs missing regions in MODIS-derived temperature maps under cloud contamination. These results demonstrate the potential of SR-Weather for enhancing machine learning-based weather forecasts at fine spatial scales.

📄 **Paper:** [npj Climate and Atmospheric Science (2026)](https://doi.org/10.1038/s41612-026-01328-5)

## Framework

![SR-Weather Framework](assets/framework.png)

## Repository Structure

```text
SR-Weather/
├── model.py
├── scm.py
├── assets/
│   └── framework.png
└── README.md
```

## Code Description

- **model.py**: Implementation of the SR-Weather neural network architecture.
- **scm.py**: Implementation of the spatial contextual information construction.

## Citation

If you find this work useful in your research, please cite the following paper:

```bibtex
@article{park2026srweather,
  title   = {A super-resolution framework for downscaling machine learning weather prediction toward 1-km air temperature},
  author  = {Park, Hyebin and Park, Seonyoung and Kang, Daehyun and Kim, Jeong-Hwan},
  journal = {npj Climate and Atmospheric Science},
  volume  = {9},
  pages   = {56},
  year    = {2026},
  doi     = {10.1038/s41612-026-01328-5},
  url     = {https://doi.org/10.1038/s41612-026-01328-5}
}
```
