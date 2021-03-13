import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from utils import CustomDataSetForNextFrame
from models import Predictor, CNN
from PIL import Image
from typing import List
import shutil
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--ngf", type=int, default=32, help="channels of generator")
parser.add_argument("--ndf", type=int, default=32, help="channels of discriminitor")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--action_dim", type=int, default=7, help="number of actions for dataset")
arg = parser.parse_args()

os.makedirs('predictorimages', exist_ok=True)
torch.manual_seed(1)    # reproducible
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_predictor(
    data_path: List,
    env_name: List,
    load_old_policy,
    old_policy_path:str=None,
    ):

    task_name = ''
    for name in env_name:
        task_name += name
        task_name += '_'

    root_path = 'trained_predictors'
    log_dir = 'trained_predictors/log/' + task_name
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
    
    os.makedirs(root_path+'/'+task_name+'/predicted_images', exist_ok=True)

    writer = SummaryWriter(log_dir)

    train_data = CustomDataSetForNextFrame(data_path, train=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=arg.batch_size, shuffle=True)

    if load_old_policy == True:
        predictor = Predictor()
        predictor = torch.load(old_policy_path).cuda()
    else:
        predictor = Predictor().cuda()

    optimizer = torch.optim.Adam(predictor.parameters(), lr=arg.lr)
    loss_func = nn.MSELoss()


    for epoch in range(arg.n_epochs):
        for step, (x, action, next_img, task_index) in enumerate(train_loader):

            task_index = task_index.reshape((task_index.shape[0]), 1)
            task = torch.LongTensor(task_index).cuda()
            task = torch.zeros(task.shape[0], arg.n_classes).cuda().scatter_(1, task, 1)

            x, action, next_img = x.cuda(), action.cuda(), next_img.cuda()
            action = torch.reshape(action, (action.shape[0], 1)).cuda()
            action = torch.zeros(action.shape[0], arg.action_dim).cuda().scatter_(1, action, 1)

            decoded = predictor(x, action, task)
            loss = loss_func(decoded, next_img)      
            optimizer.zero_grad()               
            loss.backward()                     
            optimizer.step()                    

            if step % 10 == 0:
                l = loss.data.cpu().numpy()
                print('tasks: ', len(env_name), 'Epoch: ', epoch, '| train loss: %.4f' % l)
                writer.add_scalar('loss', l, epoch*len(train_loader) + step)

        if epoch % 100 == 0:
            pair = torch.cat([next_img[0:5], decoded[0:5]], 0)
            save_image(pair.data, root_path+'/'+task_name+'/predicted_images/'+"/%d.png" % epoch, nrow=5, normalize=True)
            torch.save(predictor, root_path+'/'+task_name+'/predictor.pth')


def predict_images(
    data_path,
    policy_path,    
    predictor_path,
    envs
):
    env_names = []
    for _, folders, _ in os.walk(data_path):
        env_num = len(folders)
        env_names = envs[0:env_num]
        break

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        ]
    )

    policy = torch.load(policy_path).cuda()
    policy.eval()
    predictor = torch.load(predictor_path).cuda()
    predictor.eval()

    for i in range(1, env_num+1):
        task = Variable(torch.LongTensor((i-1) * torch.LongTensor(np.ones([1, 1])))).cuda()
        task_index = torch.zeros(task.shape[0], arg.n_classes).cuda().scatter_(1, task, 1)

        for _, folders, _ in os.walk(data_path+'/'+env_names[i-1]+'_'):
            rollout_num = len(folders)
            break
        
        for rollout in range(1, rollout_num+1):
            folder_name = data_path+'/'+env_names[i-1]+'_/rollout'+str(rollout)
            print(folder_name)

            for _,_,c in os.walk(folder_name):
                break
            for fileitem in c:
                if fileitem == '0.png':
                    pass
                else:
                    os.remove(folder_name+'/'+fileitem)

            print('generating rollout ', rollout, 'env name: ', env_names[i-1])
            actions = []
            for j in range(150):  # max number of steps
                img = Image.open(folder_name+'/'+str(j)+'.png').convert('RGB') 
                
                img = transform(img).cuda()
                img = torch.unsqueeze(img, 0)
                action = policy(img)
                loss_to_stop = nn.CrossEntropyLoss()(action, torch.Tensor([0]).long().cuda())
                print('step: ', j, 'loss: ', loss_to_stop.item())
                
                action = torch.argmax(action)
                print('step: ', j, 'action: ', action.data.cpu().numpy())

                if action == 0: 
                    action = torch.Tensor([0])[0].long().cuda()
                    actions.append(action.data.cpu().numpy())
                    actions = np.array(actions)
                    new_actions = []
                    for i in range(actions.shape[0]):
                        if actions[i] == 0:
                            new_actions.append([0, 0, 0, 0])
                        elif actions[i] == 1:
                            new_actions.append([2, 0, 0, 0])
                        elif actions[i] == 2:
                            new_actions.append([-2, 0, 0, 0])
                        elif actions[i] == 3:
                            new_actions.append([0, 2, 0, 0])
                        elif actions[i] == 4:
                            new_actions.append([0, -2, 0, 0])
                        elif actions[i] == 5:
                            new_actions.append([0, 0, 2, 0])
                        elif actions[i] == 6:
                            new_actions.append([0, 0, -2, 0])
                    actions = np.array(new_actions)

                    np.save(folder_name+'/'+ 'actions.npy', actions)
                    break

                else:
                    actions.append(action.data.cpu().numpy())

                    action = torch.reshape(action, (1, 1)).cuda()
                    action = torch.zeros(action.shape[0], arg.action_dim).cuda().scatter_(1, action, 1)
                    next_img = predictor(img, action, task_index)
                    save_image(next_img.data, folder_name+'/'+str(j+1)+'.png', nrow=1, normalize=True)
                    img = next_img
            