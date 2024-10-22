from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv

# from stable_baselines3.common.envs import DummyVecEnv
from model import USV
import numpy as np
import argparse
from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path

from control_2 import Car_Dynamics, MPC_Controller, ParticleFilter

parser = argparse.ArgumentParser()
parser.add_argument("--x_start", type=int, default=0, help="X of start")
parser.add_argument("--y_start", type=int, default=90, help="Y of start")
parser.add_argument("--psi_start", type=int, default=0, help="psi of start")
parser.add_argument("--x_end", type=int, default=90, help="X of end")
parser.add_argument("--y_end", type=int, default=80, help="Y of end")
parser.add_argument(
    "--parking", type=int, default=1, help="park position in parking1 out of 24"
)

args = parser.parse_args()

final_path = np.load("first_optimal_path.npy")
final_path = final_path*20


initial_positions = [(final_path[0][0], final_path[0][1], args.psi_start)]

env = USV(
    x=args.x_start,
    y=args.y_start,
    psi=args.psi_start,
    v=0,
    dt=0.2,
    path_index=0,
    goal=final_path[-1],
    budget=20,
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
    0.7 # Fraction of entire training period over which the exploration rate is reduced
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
model.learn(total_timesteps=100000)

# Save the model
model.save("dqn_communication_optimization_g09_epsfrac07_bs128_steps100k_turtlebot_path")
