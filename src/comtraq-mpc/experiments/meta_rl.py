import argparse

import numpy as np
from pathplanning import ParkPathPlanning, PathPlanning, interpolate_path
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn.policies import MlpPolicy

from control_2 import Car_Dynamics, MPC_Controller, ParticleFilter
from environment import Environment, Parking1

# from stable_baselines3.common.envs import DummyVecEnv
from model import USV

# from stable_baselines3.common.vec_env import DummyVecEnv


# parser = argparse.ArgumentParser()
# parser.add_argument("--x_start", type=int, default=0, help="X of start")
# parser.add_argument("--y_start", type=int, default=90, help="Y of start")
# parser.add_argument("--psi_start", type=int, default=0, help="psi of start")
# parser.add_argument("--x_end", type=int, default=90, help="X of end")
# parser.add_argument("--y_end", type=int, default=80, help="Y of end")
# parser.add_argument(
#     "--parking", type=int, default=1, help="park position in parking1 out of 24"
# )

# args = parser.parse_args()

# start = np.array([args.x_start, args.y_start])
# end = np.array([args.x_end, args.y_end])
parking1 = Parking1(1)
end, obs = parking1.generate_obstacles()


park_path_planner = ParkPathPlanning(obs)
path_planner = PathPlanning(obs)

initial_positions = [
    (0, 90, 0),
    (60, 90, 270),  # The initial default position
    # Add more positions as tuples (x, y, psi)
    # Example: (10, 90, 0), (20, 90, 0), ...
]

final_paths_list = []

for x_start, y_start, psi_start in initial_positions:

    park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)
    (
        new_end,
        park_path,
        ensure_path1,
        ensure_path2,
    ) = park_path_planner.generate_park_scenario(
        int(x_start), int(y_start), int(end[0]), int(end[1])
    )

    # print("planning park scenario ...")

    # print("routing to destination ...")
    path = path_planner.plan_path(
        int(x_start), int(y_start), int(new_end[0]), int(new_end[1])
    )
    path = np.vstack([path, ensure_path1])

    print("interpolating ...")
    interpolated_path = interpolate_path(path, sample_rate=5)
    interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    interpolated_park_path = np.vstack(
        [ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]]
    )

    final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    print(len(final_path))
    print(new_end)
    final_paths_list.append(final_path)


env = USV(
    x=initial_positions[0][0],
    y=initial_positions[0][1],
    psi=initial_positions[0][2],
    v=0,
    dt=0.2,
    path_index=0,
    goal=new_end,
    budget=20,
    initial_positions=initial_positions,
    final_paths=final_paths_list,
)
check_env(env)  # Optional: Check if the environment follows Gym API
# env.set_initial_state(initial_positions[0][0], initial_positions[0][1], initial_positions[0][2])
# vec_env = DummyVecEnv([lambda: env])
# Parameters to modify
buffer_size = 10000  # Size of the replay buffer
learning_rate = 1e-3  # Learning rate
batch_size = 128  # Size of the batch for learning
gamma = 0.9  # Discount factor
exploration_fraction = (
    7
    / 8  # Fraction of entire training period over which the exploration rate is reduced
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
model.learn(total_timesteps=400000)

# Save the model
model.save("dqn_communication_optimization_meta_400k")
