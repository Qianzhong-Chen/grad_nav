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

- install nerfstudio (refer to the [link](https://github.com/nerfstudio-project/nerfstudio)), by default, using nerfstudio conda env for future development.

- Confirm all 3DGS packages have been installed
  ```
  conda activate nerfstudio
  pip install gsplat
  ```

- Download [3DGS data](https://drive.google.com/drive/folders/1nx2JLNtK6uSJuDX8HUS75eTrn6gfR0jG?usp=sharing) and [point cloud data](https://drive.google.com/drive/folders/1nx2JLNtK6uSJuDX8HUS75eTrn6gfR0jG?usp=sharing) to
  ```
  GRaD_Nav_internal/envs/assets/
  ```

- Set up [wandb](https://docs.wandb.ai/quickstart/) or comment out related code
<!-- - dflex

  ```
  cd dflex
  pip install -e .
  ``` -->

<!-- - gym 

  ```
  pip install gym
  ``` -->

<!-- 
#### Test Examples

A test example can be found in the `examples` folder.

```
python test_env.py --env AntEnv
```

If the console outputs `Finish Successfully` in the last line, the code installation succeeds.
 -->

## Training
Checkout <.github/launch.json>, it is highly recommended to use VSCode debugger



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