import numpy as np
import time
from torch.utils.data import Dataset
from torchvision import transforms
import os
import shutil
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from typing import List
from PIL import Image
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise
from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed
from torchvision.utils import save_image
from models import Generator, Discriminator

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

transform = transforms.Compose([
    transforms.ToTensor(),
    ]
)

# gaussian noise for transform
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=10.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        if np.random.rand() > 0.5:
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, c))
            img = N + img
            img[img > 255] = 255                 
            img[img < 0] = 0
        img = img.astype('uint8')
        img = Image.fromarray(img).convert('RGB')

        return img

class AddGaussianBlur(object):
    def __init__(self, radius=7, sigma=10):

        self.radius = radius
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        if np.random.rand() > 0.3:
            img = cv2.GaussianBlur(img, (self.radius, self.radius), self.sigma)
            img[img > 255] = 255                 
            img[img < 0] = 0

        img = img.astype('uint8')
        img = Image.fromarray(img).convert('RGB')

        return img


# gan
class CustomDataSetForFirstFrame(Dataset):
    def __init__(self, paths, transform=transform, train=True):
        '''
        paths is a list of different envs:
        [
            [rollout1, rollout2, ..., rollout100],
            [rollout1, rollout2, ..., rollout100],
            ...,
            ...,
            [rollout1, rollout2, ..., rollout100],
        ]

        '''
        self.paths = paths
        self.transform = transform
        self.imgs = None

        for i in range(len(self.paths)):
            for j in range(1, 101): # total 100 rollouts in one env
                rollout_path = self.paths[i] + '/rollout' + str(j)
                img_path = rollout_path + '/0.png'
                new_img = np.array([np.array(Image.open(img_path).convert('RGB'))])
                if i + j == 1:
                    self.imgs = new_img
                else:
                    self.imgs = np.concatenate((self.imgs, new_img), axis=0)
        self.length = 100 * len(self.paths)        
     
            

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        if self.transform:
            img = self.transform(img)
        
        return img
 
    def __len__(self):
        return self.length



# bc
class CustomDataSetForAll(Dataset):
    def __init__(self, paths, transform=transform, train=True):
        '''
        paths is a list of different envs:
        [
            [rollout1, rollout2, ..., rollout100],
            [rollout1, rollout2, ..., rollout100],
            ...,
            ...,
            [rollout1, rollout2, ..., rollout100],
        ]

        '''
        self.train = train
        self.paths = paths
        self.length = 0
        self.actions = []
        self.imgs = []
        

        '''OUTPUT form is:
        [
            [img0, img1, ..., imgn],
            [img0, img1, ..., imgn],
            ...,
            [img0, img1, ..., imgn],
        ]
        '''

        for i in range(len(self.paths)):
            for j in range(1, 101): # total 100 rollouts in one env
                self.imgs.append([])
                rollout_path = self.paths[i] + '/rollout' + str(j)

                new_actions = np.load(rollout_path+'/actions.npy')
                self.actions.append(new_actions)
                            
                for _,_,c in os.walk(rollout_path):
                    img_number = len(c)-1
                    break
                self.length += img_number
                for k in range(img_number):
                    img_path = rollout_path + '/'+str(k)+'.png'
                    new_img = np.array(np.array(Image.open(img_path).convert('RGB')))
                    self.imgs[-1].append(new_img)


        self.transform = transform
        if self.train == True:
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.RandomErasing(scale=(0.03, 0.1), value='1234'),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                AddGaussianNoise(mean=0, variance=1, amplitude=5),
                AddGaussianBlur(radius=3, sigma=8),
                transforms.ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        base = 0
        for rollout in range(len(self.imgs)):
            if len(self.imgs[rollout]) + base > idx:
                pos = idx - base
                img = Image.fromarray(self.imgs[rollout][pos])
                action = self.actions[rollout][pos]
                break
            else:
                base += len(self.imgs[rollout])
        
        if self.transform:
            img = self.transform(img)

        if (action == [0, 0, 0, 0]).all():  # stop action
            action = torch.LongTensor([0])[0]
        else:
            if action[0] == 2:
                action = torch.LongTensor([1])[0]
            elif action[0] == -2:
                action = torch.LongTensor([2])[0]
            elif action[1] == 2:
                action = torch.LongTensor([3])[0]
            elif action[1] == -2:
                action = torch.LongTensor([4])[0]
            elif action[2] == 2:
                action = torch.LongTensor([5])[0]
            elif action[2] == -2:
                action = torch.LongTensor([6])[0]

        return img, action
 
    def __len__(self):
        return self.length


# predictor
class CustomDataSetForNextFrame(Dataset):
    def __init__(self, paths, transform=transform, train=True):
        self.paths = paths
        self.train = train
        self.actions = []
        self.imgs = []
        self.next_imgs = []
        self.length = 0

        for i in range(len(self.paths)):
            for j in range(1, 101): # total 100 rollouts in one env
                self.imgs.append([])
                self.next_imgs.append([])
                rollout_path = self.paths[i] + '/rollout' + str(j)

                # since we here predict next frame, we
                # only get [0:-1] actions, i.e. remove last action
                new_actions = np.load(rollout_path+'/actions.npy')[:-1]
                self.actions.append(new_actions)

                for _,_,c in os.walk(rollout_path):
                    # since we here predict next frame, we
                    # remove last frame
                    img_number = len(c)-2
                    break
                self.length += img_number
                for k in range(img_number):
                    img_path = rollout_path + '/'+str(k)+'.png'
                    new_img = np.array(Image.open(img_path).convert('RGB'))
                    next_img_path = rollout_path + '/'+str(k+1)+'.png'
                    new_next_img = np.array(Image.open(next_img_path).convert('RGB'))

                    self.imgs[-1].append(new_img)
                    self.next_imgs[-1].append(new_next_img)

        self.transform = transform
        if self.train == True:
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.RandomErasing(scale=(0.03, 0.1), value='1234'),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                AddGaussianNoise(mean=0, variance=1, amplitude=5),
                AddGaussianBlur(radius=3, sigma=8),
                transforms.ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        base = 0
        
        for rollout in range(len(self.imgs)):
            if len(self.imgs[rollout]) + base > idx:
                pos = idx - base
                img = Image.fromarray(self.imgs[rollout][pos])
                next_img = Image.fromarray(self.next_imgs[rollout][pos])
                action = self.actions[rollout][pos]

                task_index = rollout // 100
                
                break
            else:
                base += len(self.imgs[rollout])
        
        img = self.transform(img)
        next_img = transforms.ToTensor()(next_img)
        
        if (action == [0, 0, 0, 0]).all():  # stop action
            action = torch.LongTensor([0])[0]
        else:
            if action[0] == 2:
                action = torch.LongTensor([1])[0]
            elif action[0] == -2:
                action = torch.LongTensor([2])[0]
            elif action[1] == 2:
                action = torch.LongTensor([3])[0]
            elif action[1] == -2:
                action = torch.LongTensor([4])[0]
            elif action[2] == 2:
                action = torch.LongTensor([5])[0]
            elif action[2] == -2:
                action = torch.LongTensor([6])[0]
                
        return img, action, next_img, task_index
 
    def __len__(self):
        return self.length


def collect_reachv2(collect_again=False, rgb=False):
    env_name = 'reach-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomReachV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=80,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_buttonpressv2(collect_again=False, rgb=False):
    env_name = 'button-press-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomButtonPressV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=80,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_draweropenv2(collect_again=False, rgb=False):
    env_name = 'drawer-open-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomDrawerOpenV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=80,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_dooropenv2(collect_again=False, rgb=False):
    env_name = 'door-open-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomDoorOpenV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=200,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_sweepv2(collect_again=False, rgb=False):
    env_name = 'sweep-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomSweepV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=100,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_pushv2(collect_again=False, rgb=False):
    env_name = 'push-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomPushV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=100,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_sweepintov2(collect_again=False, rgb=False):
    env_name = 'sweep-into-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomSweepIntoV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=100,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_coffeebuttonv2(collect_again=False, rgb=False):
    env_name = 'coffee-button-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomCoffeeButtonV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=100,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_faucetopenv2(collect_again=False, rgb=False):
    env_name = 'faucet-open-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomFaucetOpenV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=100,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )

def collect_windowopenv2(collect_again=False, rgb=False):
    env_name = 'window-open-v2'
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = CustomWindowOpenV2Policy()
    get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=200,
                resolution=(64, 64),
                # camera='corner'
                camera='topview',
                collect_again=collect_again,
                rgb=rgb
                )



def get_rollouts(env, 
                env_name,
                policy, 
                max_rollouts=100, 
                max_step=80,
                resolution=(64, 64),
                camera='corner',
                collect_again=False,
                rgb=False
                ):
    '''
    max_step: max step in one episode, in case 
                somtimes the policy would stuck in one episode
    '''
    
    if rgb == False:
        root_path = env_name + '_' + camera
    else:
        root_path = 'rgb/' + env_name + '_' + camera

    if os.path.exists(root_path) and collect_again == False:
        print('already collected!')
        return 0

    rollouts = 0
    step=-1
    ob = env.reset()
    # env.render()

    # initiallization
    open_actions = [[0, 0, 0, 0] for _ in range(100)]
    for a in open_actions:
        ob, reward, done, info = env.step(a)
        # env.render()
        time.sleep(0.005)

    max_rollouts += 1
    while True:
        if os.path.exists(root_path + '/rollout' + str(rollouts)):
            shutil.rmtree(root_path + '/rollout' + str(rollouts))
            os.makedirs(root_path + '/rollout' + str(rollouts))
        else:
            os.makedirs(root_path + '/rollout' + str(rollouts))

        step = -1
        actions = []
        for _ in range(20):
            ob, reward, done, info = env.step([0, 0, 0, 0])
            # env.render()

        while rollouts < max_rollouts:
            step += 1
            print(step)

            if step >= max_step:
                step = -1
                ob = env.reset()
                shutil.rmtree(root_path + '/rollout' + str(rollouts))
                os.makedirs(root_path + '/rollout' + str(rollouts))
                actions = []
                continue

            action = policy.get_action(ob)
            # print(action)
            actions.append(action)

            img = env.sim.render(*resolution, mode='offscreen', camera_name=camera)[:,:,::-1]
            img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

            if rgb == False:
                cv2.imwrite(root_path + '/rollout' + str(rollouts) + '/' + str(step) + '.png', img_gray)
            else:
                cv2.imwrite(root_path + '/rollout' + str(rollouts) + '/' + str(step) + '.png', img)


            ob, reward, done, info = env.step(action)
            # env.render()
            time.sleep(0.005)
            
            for _ in range(20):
                ob, reward, done, info = env.step([0, 0, 0, 0])
                # env.render()


            if info['success']:
                actions.append(np.zeros((4)))                
                actions = np.array(actions)
                img = env.sim.render(*resolution, mode='offscreen', camera_name=camera)[:,:,::-1]
                img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                if rgb == False:
                    cv2.imwrite(root_path + '/rollout' + str(rollouts) + '/' + str(step+1) + '.png', img_gray)
                else:
                    cv2.imwrite(root_path + '/rollout' + str(rollouts) + '/' + str(step+1) + '.png', img)

                print(actions.shape)
                assert actions.shape[0] == (step+2)
                np.save(root_path + '/rollout' + str(rollouts) + '/' + 'actions.npy', actions)
                rollouts += 1
                step = -1
                ob = env.reset()
                actions = []
                break

        if rollouts >= max_rollouts:
            print('successfully collected '+str(rollouts-1)+' trails')
            shutil.rmtree(root_path + '/rollout' + str(0))
            break
        

def move(from_xyz, to_xyz):
    """Computes action components that help move from 1 position to another

    Args:
        from_xyz (np.ndarray): The coordinates to move from (usually current position)
        to_xyz (np.ndarray): The coordinates to move to
        p (float): constant to scale response

    Returns:
        (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

    """
    'first x, then y, then z'
    response = [0, 0, 0]
    error = to_xyz - from_xyz

    if abs(error[0]) >= 0.01:
        if error[0] > 0:
            response[0] = 2
        else:
            response[0] = -2
        return response
    elif abs(error[1]) >= 0.01:
        if error[1] > 0:
            response[1] = 2
        else:
            response[1] = -2
        return response
    elif abs(error[2]) >= 0.01:
        if error[2] > 0:
            response[2] = 2
        else:
            response[2] = -2
        return response
    else:
        print('no error')
        return response


class CustomReachV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'puck_pos': obs[3:6],
            'goal_pos': obs[9:],
            'unused_info': obs[6:9],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=o_d['goal_pos'])
        action['grab_effort'] = 0.

        return action.array


class CustomButtonPressV2Policy(Policy):

    @staticmethod
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'button_start_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self.desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_start_pos'] + np.array([0., 0., -0.07])

        return pos_button


class CustomDrawerOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'drwr_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        # align end effector's Z axis with drawer handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
            to_pos = pos_drwr + np.array([0., 0., 0.3])
            action['delta_pos'] = move(o_d['hand_pos'], to_pos)
        # drop down to touch drawer handle
        elif abs(pos_curr[2] - pos_drwr[2]) > 0.02:
            to_pos = pos_drwr
            action['delta_pos'] = move(o_d['hand_pos'], to_pos)
        # push toward a point just behind the drawer handle
        else:
            to_pos = pos_drwr + np.array([0., -0.06, 0.])
            action['delta_pos'] = move(o_d['hand_pos'], to_pos)

        # keep gripper open
        action['grab_effort'] = 0.

        return action.array


class CustomDoorOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'door_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        # up the arm
        if np.linalg.norm(pos_curr[:2] - [0,0.6]) < 0.01 and pos_curr[2] <= 0.21:
            return [pos_curr[0], pos_curr[1], 0.22]

        # align end effector's Z axis with door handle's Z axis
        elif np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.1:
            return pos_door + np.array([0.06, 0.02, 0.2])
        # drop down on front edge of door handle
        elif abs(pos_curr[2] - pos_door[2]) > 0.02:
            return pos_door + np.array([0.06, 0.02, 0.])
        # push from front edge toward door handle's centroid
        else:
            if pos_curr[1] >= 0.5:
                return [pos_curr[0], 0.4, pos_curr[2]]
            else:
                return pos_door


class CustomSweepV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'cube_pos': obs[3:6],
            'unused_info': obs[6:9],
            'goal_pos': obs[9:]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos'] 
        pos_goal = o_d['goal_pos']

        if pos_curr[2] >= 0.056:
            return pos_cube + [-0.05, -0.05, 0.05]
        else:
            return pos_goal


class CustomPushV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'puck_pos': obs[3:6],
            'goal_pos': obs[9:],
            'unused_info': obs[6:9],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] 
        pos_goal = o_d['goal_pos']
        print(pos_curr)
        print(pos_puck)
        if pos_curr[2] >= 0.066 and pos_curr[1] < 0.61:
            return pos_puck + [0, -0.1, 0.03]
        else:
            return pos_goal


class CustomSweepIntoV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'cube_pos': obs[3:6],
            'unused_info': obs[6:9],
            'goal_pos': obs[9:]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos']
        pos_goal = o_d['goal_pos']

        print(pos_curr)
        if pos_curr[2] >= 0.107 and pos_curr[1] < 0.62:
            return pos_cube + [0, -0.1, 0.01]
        else:
            return pos_goal



class CustomCoffeeButtonV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'button_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([.0, .0, -.07])

        if np.linalg.norm(pos_curr[[0, 2]] - pos_button[[0, 2]]) > 0.02:
            return np.array([pos_button[0], pos_curr[1], pos_button[2]])
        else:
            return pos_button + np.array([.0, .2, .0])


class CustomFaucetOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'faucet_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        
        pos_faucet = o_d['faucet_pos'] + np.array([-.04, .0, .03])
        if np.linalg.norm(pos_curr[:2] - pos_faucet[:2]) > 0.04 and pos_curr[1] < 0.62:
            return pos_faucet 
            
        elif abs(pos_curr[2] - pos_faucet[2]) > 0.02:
            
            return pos_faucet
        else:
            return pos_faucet + np.array([.02, .05, .0])


class CustomWindowOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'wndw_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d))
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_wndw = o_d['wndw_pos'] + np.array([-0.03, -0.03, -0.08])

        if np.linalg.norm(pos_curr[:2] - pos_wndw[:2]) > 0.04:
            return pos_wndw + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_wndw[2]) > 0.02:
            return pos_wndw
        else:
            return pos_wndw + np.array([0.1, 0., 0.])


def generate_images(
    env_name: List, 
    generator_path: str,
    trail_num=100,
    method='wgangp'
    ):
    task_name = ''
    for name in env_name:
        task_name += name
        task_name += '_'

    for env in env_name:
        for rollout in range(1, trail_num+1):
            os.makedirs('generated_images/'+task_name+'/'+env+'_/rollout'+str(rollout), exist_ok=True)

    latent_dim = 100

    generator = torch.load(generator_path)
    for i in range(1, trail_num+1):
        z = torch.randn(len(env_name), latent_dim, 1, 1).cuda()
        gen_imgs = generator(z)
        if os.path.exists('generated_images/'+str(task_name)+'/rollout'+str(i)):
            shutil.rmtree('generated_images/'+str(task_name)+'/rollout'+str(i))

        for n in range(len(env_name)):
            save_image(gen_imgs.data[n], 'generated_images/'+str(task_name)+'/'+str(i)+str(n)+'.png', nrow=1, normalize=True)

        print('generate '+str(i)+'th first image!')
