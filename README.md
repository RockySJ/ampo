# Model-based Policy Optimization with Unsupervised Model Adaptation

This repository contains a TensorFlow implementation of AMPO.


## Requirements

1. Install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and copy your license key to `~/.mujoco/mjkey.txt`
2. `pip install -r requirements.txt`

## Running
Configuration files can be found in [`config/`](config).
For AMPO with Wasserstein distance
```
python main.py --config=config.ant
```
For AMPO with Maximum Mean Discrepancy
```
python main.py --config=config.ant_mmd
```


## Contact
Please email to rockyshen@apex.sjtu.edu.cn should you have any questions, comments or suggestions.

## Acknowledgments
The code implementation is mainly modified based on the [MBPO](https://github.com/JannerM/mbpo) codebase.
