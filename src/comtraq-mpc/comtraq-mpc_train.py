import argparse

import numpy as np

# from pathplanning import ParkPathPlanning, PathPlanning, interpolate_path
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn.policies import MlpPolicy

from control_2 import Car_Dynamics, MPC_Controller, ParticleFilter
from environment import Environment, Parking1

# from stable_baselines3.common.envs import DummyVecEnv
from model import USV

# from stable_baselines3.common.vec_env import DummyVecEnv


final_path = np.genfromtxt(
    "dense_lawnmower_pattern_200_waypoints.csv", delimiter=",", skip_header=1
)
# final_path = final_path[::10]
# final_path[:,:2]*=15


initial_positions = [(final_path[0][0], final_path[0][1], 90.0)]
# print(initial_positions)
final_path = final_path[:, :2]


env = USV(
    v=0,
    dt=0.2,
    path_index=0,
    goal=final_path[-1],
    budget=50,
    initial_positions=initial_positions,
    final_paths=[final_path],
)
check_env(env)  # Optional: Check if the environment follows Gym API
# env.set_initial_state(args.x_start, args.y_start, args.psi_start)
# vec_env = DummyVecEnv([lambda: env])
# Parameters to modify
buffer_size = 10000  # Size of the replay buffer
learning_rate = 1e-3  # Learning rate
batch_size = 128  # Size of the batch for learning
gamma = 0.9  # Discount factor
exploration_fraction = (
    0.7  # Fraction of entire training period over which the exploration rate is reduced
)
exploration_final_eps = 0.02  # Final value of random action probability
target_update_interval = (
    1000  # Number of steps after which the target network is updated
)
train_freq = (1, "episode")  # Update the model every 'train_freq' steps
gradient_steps = -1  # Number of gradient steps to take after each environment step
log_dir = "tmp/dqn/"  # Where to log the model
# Initialize the model with custom parameters
model = DQN(
    MlpPolicy,
    env,
    buffer_size=buffer_size,
    learning_rate=learning_rate,
    batch_size=batch_size,
    gamma=gamma,
    exploration_fraction=exploration_fraction,
    exploration_final_eps=exploration_final_eps,
    target_update_interval=target_update_interval,
    train_freq=train_freq,
    gradient_steps=gradient_steps,
    verbose=1,
    tensorboard_log=log_dir,
)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save(
    "dqn_communication_optimization_epsfrac07_steps1M_50bud_200_waypoints_lawnmower_path"
)
