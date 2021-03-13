# CRIL
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This is the official repository for the implemntation of simulation experiments of CRIL: Continual Robot Imitation Learning via Generative Dynamics Model by [Chongkai Gao](http://chongkaigao.com/), Haichuan Gao, Shangqi Guo, Tianren Zhang and Feng Chen.

__Table of Contents__
- [Introduction](#introduction)
- [Installation](#installation)
- [Running](#running)
- [Citing CRIL](#citing-cril)
- [Acknowledgements](#acknowledgements)

## Introduction

CRIL is a specialized deep generative replay algorithm designed for continual robot imitation learning that employs both a dynamics predictor and WGAN-GP for trajectory replay. The results of simulation and realworld experiments are as follows:

<div align=center><img src="https://github.com/HeegerGao/CRIL/blob/main/pictures/res1.png" width="322" height="292" alt="res1"/>&nbsp&nbsp<img src="https://github.com/HeegerGao/CRIL/blob/main/pictures/res2.png" width="322" height="292" alt="res2"/></div>

The replayed images of CRIL are as follows:

<div align=center><img src="https://github.com/HeegerGao/CRIL/blob/main/pictures/CRIL.png" width="635" height="414" alt="CRIL"/></div>




## Installation
The simulation experiments of CRIL are based on MuJoCo and Meta-World benchmark, which need to be installed in advance. You can follow these instructions to install [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco) and [meta-world](https://github.com/rlworkgroup/metaworld).


## Running

Run the following code to train the models:
`python3 main.py`

Note: you cannot run with only one click for various reasons. See the instructions in [main.py](https://github.com/HeegerGao/CRIL/blob/main/main.py).


## Citing-CRIL

## Acknowledgements
We would like to thank Xin Su, Zhile Yang and Yizhou
Jiang for various discussions on DGR theory and experiments
of GANs. This work was supported in part by the National
Natural Science Foundation of China under Grant 61671266
and Grant 61836004, in part by the Tsinghua-Guoqiang
research program under Grant 2019GQG0006, and in part by
Qualcomm Technologies, Inc.
