# GRaD-Nav

This repository contains the implementation for the paper [GRaD-Nav: Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics](https://qianzhong-chen.github.io/gradnav.github.io/).

<!-- 

In this paper, we present a GPU-based differentiable simulation and propose a policy learning method named SHAC leveraging the developed differentiable simulation. We provide a comprehensive benchmark set for policy learning with differentiable simulation. The benchmark set contains six robotic control problems for now as shown in the figure below. 

<p align="center">
    <img src="figures/envs.png" alt="envs" width="800" />
</p> -->

## Installation

- `git clone git@github.com:Qianzhong-Chen/GRaD_Nav_internal.git`


#### Prerequisites

- Configure [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), by default, using nerfstudio conda env for future development.

- Confirm all 3DGS packages have been installed
  ```
  conda activate nerfstudio
  pip install gsplat
  ```

## Data Download and Setup

1. **Download Required Data:**
   - [3DGS Data](https://drive.google.com/drive/folders/1nx2JLNtK6uSJuDX8HUS75eTrn6gfR0jG?usp=sharing)
   - [Point Cloud Data](https://drive.google.com/drive/folders/1nx2JLNtK6uSJuDX8HUS75eTrn6gfR0jG?usp=sharing)

2. **Place the downloaded folders in:**
  <GRaD_Nav_internal/envs/assets/>



3. **Update Configuration:**
- Modify lines **4-14** in:
  ```
  GRaD_Nav_internal/envs/assets/gs_data/map_name/splatfacto/time/config.yml
  ```
- Set the corresponding paths to your **3DGS data folder** manually.

## Create a new branch and check out

```
git branch <branch_name>
git checkout <branch_name>
```

## Training
- Checkout <.github/launch.json>, it is highly recommended to use VSCode debugger
- Set up [wandb](https://docs.wandb.ai/quickstart/) (highly recommended) or comment out related code


## Testing
Same as above.


## Citation

If you find our paper or code is useful, please consider citing:
```kvk
  @misc{chen2025gradnavefficientlylearningvisual,
        title={GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics}, 
        author={Qianzhong Chen and Jiankai Sun and Naixiang Gao and JunEn Low and Timothy Chen and Mac Schwager},
        year={2025},
        eprint={2503.03984},
        archivePrefix={arXiv},
        primaryClass={cs.RO},
        url={https://arxiv.org/abs/2503.03984}, 
    }
```