# ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates

[Gokul Puthumanaillam](https://gokulp01.github.io/)<sup>1*</sup>,
[Manav Vora](https://manavvora.github.io)<sup>1*</sup>,
[Melkior Ornik](https://vita-group.github.io/)<sup>1</sup>

<sup>1</sup>University of Illinois at Urbana-Champaign

<sup>\*</sup> denotes equal contribution.

[Paper](https://arxiv.org/pdf/2403.01564) | Video (Coming soon)

$\text{\color{red}{News\!}}$ ComTraQ-MPC is accepted at _IROS 2024_ 🎉.

[Here](https://youtu.be/mqOYaBQ2wVI?t=1647) is a very quick overview of our work being presented by Prof. Ornik at Purdue University.

---

## Introduction

Optimal decision-making for trajectory tracking in partially observable, stochastic environments where the number of active localization updates—the process by which the agent obtains its true state information from the sensors are limited, presents a significant challenge. Traditional methods often struggle to balance resource conservation, accurate state estimation and precise tracking, resulting in suboptimal performance. This problem is particularly pronounced in environments with large action spaces, where the need for frequent, accurate state data is paramount, yet the capacity for active localization updates is restricted by external limitations. This paper introduces ComTraQ-MPC, a novel framework that combines Deep Q-Networks (DQN) and Model Predictive Control (MPC) to optimize trajectory tracking with constrained active localization updates. The meta- trained DQN ensures adaptive active localization scheduling, while the MPC leverages available state information to improve tracking. The central contribution of this work is their reciprocal interaction: DQN’s update decisions inform MPC’s control strategy, and MPC’s outcomes refine DQN’s learning, creating a cohesive, adaptive system.

## Installation

### Clone the repository

Clone the repository to your local machine using the following command:

```bash
git clone git@github.com:gokulp01/ComTraq-MPC.git
cd ComTraQ-MPC
```

# Repository Structure

<details>
<summary>Click to expand/collapse structure</summary>

```bash
.
├── comtraq-mpc_hardware/       # Hardware implementation
│   ├── first_optimal_path_with_yaw.npy
│   ├── map_talbot_new.pgm
│   ├── test/
│   ├── turtlebot3/
│   └── turtlebot_positions.csv
├── g++check.cpp                # C++ code to check the installation
├── gitter.sh
├── logs/                       # Logs for the experiments
│   ├── asdimage.png
│   ├── fgimage.png
│   ├── log results/
│   ├── maps/
│   └── path_comparisons/
├── model_generation/           # Model generation code (in C++)
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── build/
│   ├── build_model.sh
│   ├── config.cpp
│   ├── config.h
│   ├── legacy_model_test.cpp
│   ├── main.cpp
│   ├── model/
│   ├── model.cpp
│   ├── model_demo.py
│   ├── model_legacy.cpp
│   ├── model_new.cpp
│   ├── uuv.cpp
│   └── uuv.h
├── setup.sh                    # Setup script to install required packages
└── src/                        # Source code (Python)
    ├── __init__.py
    ├── __pycache__/
    ├── baselines/              # Baselines (includes README.md for each)
    ├── comtraq-mpc/            # Main code
    ├── data_generators/        # Data generators
    ├── tmp/                    # Temporary files
    └── unit_tests/             # Unit tests
```

</details>

## Key Components

- `comtraq-mpc_hardware/`: Contains hardware implementation files.
- `logs/`: Stores experiment logs and related images.
- `model_generation/`: C++ code for model generation.
- `src/`: Main Python source code, including baselines and tests.
- `setup.sh`: Script for setting up the required environment.

### Automatic Installation

You can simply run the following command to directly install the required packages and run the code.

```bash
chmod +x setup.sh
./setup.sh
```

This will create and install the conda environment to run the code.

### Manual Installation

Follow these steps to set up the conda environment with all required packages:

1. Create a new conda environment:

   ```bash
   conda create -n myenv python=3.10 # this has been tested with every python version>=3.10
   ```

   Replace `myenv` with your preferred environment name.

2. Activate the environment:

   ```bash
   conda activate myenv
   ```

3. Install the required packages:

```bash
conda install -c conda-forge stable-baselines3 pytorch torchvision torchaudio matplotlib numpy ipykernel scipy seaborn scikit-learn -y
```

4. Verify the installation:

   ```bash
   conda list
   ```

   This will display all installed packages in the current environment.

5. To use this environment in Jupyter Notebook, add it as a kernel:
   ```bash
   python -m ipykernel install --user --name=myenv
   ```
   Replace `myenv` with the name you chose for your environment.

## Getting Started (Inference)

First-time running will take a longer time to compile the models.
This will run the Comtraq-MPC agent on the model defined in the paper. You can change the environment by changing the `model.py` file.

```bash
# 1. Script
python comtraq-mpc_test.py

# 2. You should be able to see the path tracked by the agent

# 3. Jupyter Notebook
comtraq-mpc_test.ipynb
```

## Training Your Own Model

### Data Preparation

Prepare the training data similar to the data provided in the `data` directory. The data should be in the form of a csv/npy with the following keys:

- x_pos: x position of the agent
- y_pos: y position of the agent
- z_pos: z position of the agent (optional)
- yaw: yaw of the agent (optional)

### Creating the model

Create the model.py file which defines your dynamics model. We follow gymnasium's environment API. The model should have the following functions:

- `reset`: Reset the environment
- `step`: Take a step in the environment
- `get_state`: Get the current state of the environment
- `get_action`: Get the action space of the environment
- `get_observation`: Get the observation space of the environment
- `get_reward`: Get the reward of the environment
- `get_done`: Get the done status of the environment
- `get_info`: Get the info of the environment
- `render`: Render the environment
- `close`: Close the environment

(of course, you can add more functions as needed. You can also follow the model.py file given in the repository)

Your file structure should look like this:

```
# comtraq-mpc is your base folder used in the previous steps

.
├── README.md
├── comtraq-mpc_test.ipynb
├── comtraq-mpc_test.py
├── comtraq-mpc_train.py
├── config.py
├── control.py
├── data
├── environment.py
├── experiments
├── meta_dqn_mpc.py
├── model.py
├── trained_models
└── utils.py

```

### Training

The `comtraq-mpc_train.py` script will train the model on the trajectory provided in the `data` directory.

Specify your variables in the `config.py` file. You can change the model, environment, and other hyperparameters in this file.

We use stable-baselines3 for training the model (this can be changed in the `comtraq-mpc_train.py` file). The training script will save the model in the `trained_models` directory.

Experiment logs and checkpoints will be saved in the same directory.

You can infer the model using the `comtraq-mpc_test.py` script.

### Baselines

Please follow the instructions in the `src/baselines/<baseline_name>` directory to train and infer the baseline models.

## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

```
@article{puthumanaillamvora2024comtraq,
  title={ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates},
  author={Puthumanaillam, Gokul and Vora, Manav and Ornik, Melkior},
  journal={arXiv preprint arXiv:2403.01564},
  year={2024},
  note={To be presented at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)}
}
```

> :warning: Feel free to open an issue if you find bugs or have issues running the code.

To Do:

- [ ] Add the video link
