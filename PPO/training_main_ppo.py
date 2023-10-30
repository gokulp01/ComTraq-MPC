from __future__ import absolute_import
from __future__ import print_function
import model
import os
import datetime
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt

from training_simulation_ppo import Simulation_PPO
from rollout_buffer import RolloutBuffer
from model_ppo import TrainModel_PPO
from utils_ppo import import_train_configuration, set_sumo, set_train_path


def run_ppo_train():

    config = import_train_configuration(config_file='ppo/training_settings_ppo.ini')
    path = set_train_path(config['models_path_name'])

    # Model = TrainModel_PPO(
    #     config['learning_rate_actor'], 
    #     config['learning_rate_critic'],
    #     input_dim=config['num_states'], 
    #     output_dim=config['num_actions'],
    #     eps_clip=config['eps_clip']
    # )

    # Buffer = RolloutBuffer()

    # TrafficGen = TrafficGenerator(
    #     config['max_steps'], 
    #     config['n_cars_generated']
    # )

    
        
    # Simulation = Simulation_PPO(
    #     Model,
    #     config['opt_epochs'],
    #     Buffer,
    #     TrafficGen,
    #     sumo_cmd,
    #     config['gamma'],
    #     config['max_steps'],
    #     config['green_duration'],
    #     config['yellow_duration'],
    #     config['num_states'],
    #     config['num_actions']
    # )
    
    counter = 0
    timestamp_start = datetime.datetime.now()
    num_runs = 1
    reward_multiple_runs = np.zeros((num_runs,config['total_episodes']))
    total_steps_multiple_runs=np.zeros((num_runs,config['total_episodes']))
   
    

    for n_run in range(num_runs):
        Model = TrainModel_PPO(
        config['learning_rate_actor'], 
        config['learning_rate_critic'],
        input_dim=config['num_states'], 
        output_dim=config['num_actions'],
        eps_clip=config['eps_clip']
        )

        Buffer = RolloutBuffer()

        

        Simulation = Simulation_PPO(
        Model,
        config['opt_epochs'],
        config['update_policy_timestep'],
        Buffer,
        config['gamma'],
        config['max_steps'],
        config['num_states'],
        config['num_actions']
        )
        
        print(f"Starting run number {n_run+1}")
        print("-"*25)
        episode = 0
        while episode < config['total_episodes']:
            print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
            simulation_time, episode_reward, episode_steps = Simulation.run(episode)  # run the simulation
            print(episode_reward[episode])
            reward_multiple_runs[n_run,episode] = episode_reward[episode]
            total_steps_multiple_runs[n_run,episode]=episode_steps[episode]
    
            print('Simulation time:', simulation_time, 's') #- Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
            episode += 1
            counter += 1
    
    print(reward_multiple_runs)
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_actor_model(path)

    # copyfile(src='ppo/training_settings_ppo.ini', dst=os.path.join(path, 'ppo/training_settings_ppo.ini'))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(reward_multiple_runs[0,:])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(total_steps_multiple_runs[0,:])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')

    plt.tight_layout()
    plt.savefig('training_results.png')  # Save the figure as a file
    plt.show()


    return reward_multiple_runs, return_multiple_runs, delay_multiple_runs, queue_length_multiple_runs