import os
import shutil
import torch
import torch.nn as nn
import time
from utils import CustomDataSetForAll
from models import CNN
from torch.utils.data import DataLoader
from typing import List
from PIL import Image
from torch.utils.tensorboard import SummaryWriter   

def behavior_cloning(
    data_path: List,
    env_name: List,
    test_data_path,
    batch_size=1024,
    lr=1e-3,
    max_epoch=800,
    load_old_policy=False,
    old_policy_path=None,
    what=None,
    seed=0,
):
    task_name = ''
    for name in env_name:
        task_name += name
        task_name += '_'

    if what == 'baseline':
        root_path = 'baseline/trained_policies/' + str(seed) + '/' + task_name
        log_dir = root_path
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            os.makedirs(log_dir, exist_ok=True)
            
    elif what == 'finetune':
        root_path = 'finetune/' + str(seed) + '/' + task_name
        log_dir = root_path
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            os.makedirs(log_dir, exist_ok=True)

    elif what == 'normal':
        root_path = 'trained_policies/' + str(seed) + '/' + task_name
        log_dir = root_path
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            os.makedirs(log_dir, exist_ok=True)


    
    if len(test_data_path) == 0:
        test_data_path = data_path
    
    writer = SummaryWriter(log_dir)

    train_data = CustomDataSetForAll(data_path, train=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if load_old_policy == True:
        policy = CNN()
        policy = torch.load(old_policy_path).cuda()
    else:
        policy = CNN().cuda()
    opt = torch.optim.Adam(policy.parameters(),lr=lr)
    loss_func = nn.CrossEntropyLoss()

    
    for ep in range(max_epoch):
        print('Epoch: ', ep)
        num = 0
        allloss = 0
        for img, label in train_loader:
            img = img.cuda().float()
            label = label.cuda()
            pred = policy(img)
            loss = loss_func(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            num += 1
            allloss += loss

        if ep % 10 == 0:
            torch.save(policy, root_path+'/policy_'+task_name+'.pth')
            correct_ratio = bctest(
                root_path+'/policy_'+task_name+'.pth',
                test_data_path,
                ep=ep
            )

            writer.add_scalar('correct_ratio', correct_ratio, ep)
            
        writer.add_scalar('train loss', allloss.item() / num, ep)
        
# for test
def bctest(
    policy_path: str,
    data_path: List,
    ep: int,
):
    train_data = CustomDataSetForAll(data_path, train=False)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    policy = CNN()
    policy = torch.load(policy_path).cuda()
    print(policy_path)

    correct = 0
    total = 0
    for img, label in train_loader:
        if total >= 10000:  
            break
        img = img.cuda().float()
        label = label.cuda()
        # torch.unsqueeze(img, 1)
        pred = policy(img)

        correct += sum(torch.argmax(pred, dim=1).data.cpu().numpy() == label.data.cpu().numpy())
        total += img.shape[0]
    
    print('Epoch: ', ep, ' correct ratio: ', correct / total)
    return correct / total
