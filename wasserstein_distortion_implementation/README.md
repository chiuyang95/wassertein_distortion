This repository contains codes for metamer reconstruction task with Wasserstein distortion (Experiment 4) in [Wasserstein Distortion: Unifying Fidelity and Realism](https://arxiv.org/abs/2310.03629). The experiment aims to reconstruct an image that is close to the source image under Wasserstein distortion. For details, please refer to the paper.

To execute, run `python3 wass_dist_metamer_generation.py 'PATH_TO_SOURCE_IMAGE'`.

A brief explanation for each file:
- `wass_dist_metamer_generation.py` contains the main code.
- `vgg19_n.py` generates the VGG-19 network, which is used as feature space.
- `vgg19_norm_weights.h5` contains a particular set of saved weights for the VGG-19 network, as described in the paper and first seen in [[Gatys et al., 2015]](https://proceedings.neurips.cc/paper/2015/hash/a5e00132373a7031000fd987a3c9f87b-Abstract.html).
