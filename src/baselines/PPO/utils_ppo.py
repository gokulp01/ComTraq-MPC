import configparser
# from sumolib import checkBinary
import os
import sys


def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['learning_rate_actor'] = content['model'].getfloat('learning_rate_actor')
    config['learning_rate_critic'] = content['model'].getfloat('learning_rate_critic')
    config['eps_clip'] = content['model'].getfloat('eps_clip')
    config['opt_epochs'] = content['model'].getint('opt_epochs')
    config['update_policy_timestep'] = content['model'].getint('update_policy_timestep')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['models_path_name'] = content['dir']['models_path_name']
    return config




def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 

