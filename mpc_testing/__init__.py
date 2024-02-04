from gymnasium.envs.registration import register

register(
    id="model",
    entry_point="USV",
    max_episode_steps=300,
)
