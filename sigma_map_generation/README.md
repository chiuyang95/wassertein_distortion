This repository contains the code for [Wasserstein Distortion with Intrinsic $\sigma$-map](https://openreview.net/forum?id=8lwDe1eOTV). The framework for Wasserstein distortion requires a controlling parameter $\sigma$ for each pixel location. The algorithm presented here generates $\sigma$ for every pixel automatically for arbitrary inupt image, thus eliminating dependency of either external inputs like a saliency map, or manual tuning.

To execute, run `python3 sigma_map_gen.py 'NAME_OF_SOURCE_IMAGE'` (e.g., `python3 sigma_map_gen.py 'zebra'`)

This code is tested on Python 3.10 + TensorFlow 2.11