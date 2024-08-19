# ComTraQ-MPC

We have provided all the trained models in the `trained_models` directory.
The trained model follows the following naming convention: `dqn_communication_optimization_epsfrac<epsilon_fraction>_steps<num_of_training_steps>_<map_name>.zip`.

We have also provided the ground truth trajectory for the trained model in the `data` directory.

To train your own model, you can edit the `comtraq-mpc_train.py` file and run it using the following command:

```bash
python comtraq-mpc_train.py
```

The model configuration can be changed in the `config.py` file.

## Installation

To run Comtraq-MPC, make sure you have all the dependencies installed. You can install them by following the instructions in the README.md file in the root directory of the repository.

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

Experiment logs and checkpoints will be saved in the same directory.

You can infer the model using the `comtraq-mpc_test.py` script.
