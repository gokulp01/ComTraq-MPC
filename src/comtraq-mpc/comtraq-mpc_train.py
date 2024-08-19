import config
from model import USV
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn.policies import MlpPolicy

final_path = config.final_path
final_path *= 15  # scaling the path -- can be removed if not necessary


initial_positions = [(final_path[0][0], final_path[0][1], final_path[0][2])]
final_path = final_path[:, :2]


env = USV(
    v=config.v,
    dt=config.dt,
    path_index=config.path_index,
    goal=config.goal,
    budget=config.budget,
    initial_positions=initial_positions,
    final_paths=[final_path],
)
check_env(env)
buffer_size = config.buffer_size
learning_rate = config.learning_rate
batch_size = config.batch_size
gamma = config.gamma
exploration_fraction = config.exploration_fraction
exploration_final_eps = config.exploration_final_eps
target_update_interval = config.target_update_interval
train_freq = config.train_freq
gradient_steps = config.gradient_steps
log_dir = config.log_dir
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


model.learn(total_timesteps=config.total_timesteps)

model.save(config.model_path)
