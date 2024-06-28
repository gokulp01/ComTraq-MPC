from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from model import USV
import numpy as np

# Load your environment setup as before
final_path = np.genfromtxt("turtlebot_positions.csv", delimiter=",", skip_header=1)
final_path = final_path[::10]
final_path[:,:2] *= 15

initial_positions = [(final_path[0][0], final_path[0][1], final_path[0][2])]
final_path = final_path[:,:2]

env = USV(
    v=0,
    dt=0.2,
    path_index=0,
    goal=final_path[-1],
    budget=20,
    initial_positions=initial_positions,
    final_paths=[final_path],
)
check_env(env)  # Check if the environment follows Gym API

# PPO specific parameters
learning_rate = 0.0001  # Learning rate is usually lower in PPO
n_steps = 2048  # Number of steps to run for each environment per update
batch_size = 128  # Batch size for learning
n_epochs = 10  # Number of epochs when optimizing the surrogate loss
gamma = 0.9  # Discount factor
clip_range = 0.2  # Clipping parameter, helps bounds the policy update
ent_coef = 0.0  # Entropy coefficient for exploration
log_dir = "tmp/ppo/"  # Where to log the model

# Initialize the PPO model
model = PPO(
    MlpPolicy,
    env,
    learning_rate=learning_rate,
    n_steps=n_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    gamma=gamma,
    clip_range=clip_range,
    ent_coef=ent_coef,
    verbose=1,
    tensorboard_log=log_dir,
)

# Train the model
model.learn(total_timesteps=10000000)

# Save the model
model.save("ppo_communication_optimization_epsfrac07_steps10M_turtlebot_path_baseline")
