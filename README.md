# SR-Weather🌤️

**A super-resolution framework for downscaling machine learning weather prediction toward 1-km air temperature**

[![Paper](https://img.shields.io/badge/Paper-npj%20Climate%20and%20Atmospheric%20Science-blue)](https://doi.org/10.1038/s41612-026-01328-5)
![Python](https://img.shields.io/badge/Python-3.9.12-blue)

## Abstract

> Artificial intelligence has improved the accuracy and efficiency of weather forecasting, surpassing traditional numerical weather prediction models. However, the coarse spatial resolution of global weather forecasting systems limits their ability to capture fine-scale surface heterogeneity and localized extremes, particularly in regions with complex terrain or urban heat island effects. Here, we introduce SR-Weather, a deep learning-based super-resolution framework that converts coarse 0.25° forecasts into 1-km surface air temperature fields using MODIS-derived temperature targets and high-resolution auxiliary inputs. SR-Weather outperforms existing super-resolution methods by explicitly incorporating spatial context, such as topography, impervious surface fraction, and seasonal climatology maps of air temperature. When SR-Weather was applied to the FuXi global weather forecast, the 7-day forecast error in South Korea decreased by more than 20%, which was comparable to the 1-day forecast error from low-resolution prediction using simple spatial interpolation. In addition, SR-Weather effectively reconstructs missing pixels in MODIS-derived air temperature maps under heavy cloud contamination by leveraging auxiliary variables and climatologically smoothed fields. Although validated over South Korea, the framework relies on globally available MODIS products and minimal auxiliary inputs, making it feasible to retrain for other regions. These results indicate that SR-Weather is a scalable and high-fidelity tool for enhancing machine learning-based weather forecasts at fine spatial scales.

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

## SCM Usage

The SCM is constructed from multiple yearly air-temperature datasets stored as NumPy (`.npy`) files.

### Input Data

- Each `.npy` file should contain one year of daily air-temperature data.
- Expected shape: `(365, 1, height, width)`
- All input files must have the same shape.
- Zero values are treated as missing data. Modify this behavior if zero is a valid value in your dataset.
- Place all yearly `.npy` files in a single directory.

## Citation

If you find this work useful in your research, please cite the following paper:

```bibtex
@article{park2026srweather,
  title     = {A super-resolution framework for downscaling machine learning weather prediction toward 1-km air temperature},
  author    = {Park, Hyebin and Park, Seonyoung and Kang, Daehyun and Kim, Jeong-Hwan},
  journal   = {npj Climate and Atmospheric Science},
  volume    = {9},
  articleno = {56},
  year      = {2026},
  doi       = {10.1038/s41612-026-01328-5},
  url       = {https://doi.org/10.1038/s41612-026-01328-5}
}
```
