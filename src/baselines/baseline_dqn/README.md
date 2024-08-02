# DQN Baseline

To run the baseline, make sure you have all the dependencies installed. You can install them by following the instructions in the README.md file in the root directory of the repository.

To run the baseline (inference), you can use the following command:

```bash
python baseline_dqn_test.py or use the provided notebook: baseline_dqn_test.ipynb
```

This will run the DQN agent on the model defined in the paper. You can change the environment by changing the `model.py` file.

We have provided the trained model in the `trained_models` directory.
The trained model follows the following naming convention: `dqn_communication_optimization_epsfrac<epsilon_fraction>_steps<num_of_training_steps>_<map_name>.zip`.

We have also provided the ground truth trajectory for the trained model in the `ground_truth` directory.

To train your own model, you can edit the `dqn_train.py` file and run it using the following command:

```bash
python baseline_dqn_train.py
```
