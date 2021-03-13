import torch
import torch.nn as nn
import numpy as np
import random
from utils import CustomDataSetForAll
from torch.utils.data import DataLoader
from behaviorcloning import behavior_cloning
from wgangp import train_wgangp
from predictor import train_predictor, predict_images
from utils import generate_images

envs = [
    'reach-v2',
    'button-press-v2',
    'drawer-open-v2',
    'door-open-v2',
    'sweep-v2', 
    'push-v2', 
    'sweep-into-v2',
    'coffee-button-v2',
    'faucet-open-v2',
    'window-open-v2'
]

raw_data_paths = [
    'reach-v2_topview',
    'button-press-v2_topview',
    'drawer-open-v2_topview',
    'door-open-v2_topview',
    'sweep-v2_topview', 
    'push-v2_topview',
    'sweep-into-v2_topview', 
    'coffee-button-v2_topview',
    'faucet-open-v2_topview',
    'window-open-v2_topview'
]

generated_data_paths = [
    'generated_images/reach-v2_',
    'generated_images/reach-v2_button-press-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_sweep-v2_', 
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_sweep-v2_push-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_sweep-v2_push-v2_sweep-into-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_sweep-v2_push-v2_sweep-into-v2_coffee-button-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_sweep-v2_push-v2_sweep-into-v2_coffee-button-v2_faucet-open-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_door-open-v2_sweep-v2_push-v2_sweep-into-v2_coffee-button-v2_faucet-open-v2_window-open-v2_'
]


def perform(
    task: int,
    what: str,
    seed=0,
):
    current_raw_data_paths = raw_data_paths[0:task+1]

    if task == 0:
        current_pesudo_data_paths = current_raw_data_paths
    else:
        current_pesudo_data_paths = []
        for i in range(task):
            current_pesudo_data_paths.append(generated_data_paths[task-1]+'/'+envs[i]+'_')
        current_pesudo_data_paths.append(raw_data_paths[task])

    current_envs = envs[0:task+1]

    if what == 'train_policy':
        if task == 0:
            load_old_policy = False
        else:
            load_old_policy = True
        
        old_policy_path = 'trained_policies/' + str(seed) + '/'
        task_name = ''
        for j in range(task):
            task_name += (envs[j]+'_')
        old_policy_path += task_name
        old_policy_path += '/policy_'
        old_policy_path += task_name
        old_policy_path += '.pth'

        behavior_cloning(
            current_pesudo_data_paths,
            current_envs,
            test_data_path=current_raw_data_paths,
            batch_size=1024,
            max_epoch=800,
            what='normal',
            load_old_policy=load_old_policy,
            old_policy_path=old_policy_path,
            seed=seed
        )

    elif what == 'train_generator':
        if task == 0:
            load_old_policy = False
        else:
            load_old_policy = True

        ''' You need to specify old_g_path and old_d_path every time you run this function
        '''        
        train_wgangp(
            current_pesudo_data_paths,
            current_envs,
            load_old_models=load_old_policy,
            old_g_path='trained_generators/reach-v2_/G73000.pth',
            old_d_path='trained_generators/reach-v2_/D73000.pth',
        )

    elif what == 'train_predictor':
        if task == 0:
            load_old_policy = False
        else:
            load_old_policy = True
        
        predictor_path = 'trained_predictors/'
        for j in range(task):
            predictor_path += (envs[j]+'_')
        predictor_path += '/predictor.pth'

        train_predictor(
            current_pesudo_data_paths,
            current_envs,
            load_old_policy=load_old_policy,
            old_policy_path=predictor_path,
        )
        
    elif what == 'generate_first_frames':
        ''' You need to specify old_g_path and old_d_path every time you run this function
        '''        
        generate_images(
            current_envs,
            generator_path='trained_generators/reach-v2_/G11000.pth',
            trail_num=100,
            method='wgangp'
        )

    elif what == 'predict_frames':
        policy_path = 'trained_policies/' + str(seed) + '/'
        task_name = ''
        for j in range(task+1):
            task_name += (envs[j]+'_')
        policy_path += task_name
        policy_path += '/policy_'
        policy_path += task_name
        policy_path += '.pth'

        img_path = 'generated_images/'
        predictor_path = 'trained_predictors/'
        for j in range(task+1):
            img_path += (envs[j]+'_')
            predictor_path += (envs[j]+'_')
        predictor_path += '/predictor.pth'

        predict_images(
            img_path,
            policy_path,
            predictor_path,
            envs
        )

    elif what == 'baseline':
        # using all tasks data to train
        if task == 0:
            load_old_policy = False
        else:
            load_old_policy = True
        
        old_policy_path = 'baseline/trained_policies/' + str(seed) + '/'
        task_name = ''
        for j in range(task):
            task_name += (envs[j]+'_')
        old_policy_path += task_name
        old_policy_path += '/policy_'
        old_policy_path += task_name
        old_policy_path += '.pth'

        behavior_cloning(
            current_raw_data_paths,
            current_envs,
            test_data_path=current_raw_data_paths,
            batch_size=1024,
            what='baseline',
            load_old_policy=load_old_policy,
            old_policy_path=old_policy_path,
            seed=seed,
        )

    elif what == 'finetune':
        # only use new task data to train
        if task == 0:
            load_old_policy = False
        else:
            load_old_policy = True
        
        old_policy_path = 'finetune/' + str(seed) + '/'
        task_name = ''
        for j in range(task):
            task_name += (envs[j]+'_')
        old_policy_path += task_name
        old_policy_path += '/policy_'
        old_policy_path += task_name
        old_policy_path += '.pth'


        behavior_cloning(
            [current_raw_data_paths[task]],
            current_envs,
            test_data_path=current_raw_data_paths,
            batch_size=1024,
            load_old_policy=load_old_policy,
            old_policy_path=old_policy_path,
            what='finetune',
            seed=seed,
        )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    '''
        For the following reasons, we couldn't directly 
        run the training codes with 5 seeds directly:
            1. The generator's performance is not monotone increasing
                with the training process in each task, so we need to
                manually pick the best model during the training;
            2. The generated and predicted images are not classified to
                different tasks, and we need to manually divide them during training
        Thus, we can not build an one-click running program. We need to manually
        stop and run the code for a lot of times.
    '''

    for seed in [11]:   # seeds used in the paper are: 11, 121, 111, 45, 999

        setup_seed(seed)
        for task in range(len(envs)):
            print('=====================================================')
            print('seed: ', seed, ' task: ', task)
            perform(
                task=task,

                what='train_policy',
                # what='train_generator',
                # what='train_predictor',
                # what='generate_first_frames',
                # what='predict_frames',
                # what='baseline',
                # what='finetune',

                seed=seed,
            )

