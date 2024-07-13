import os
import argparse
import sys
import data
import models
import utils
import json
import sharpness
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

from sgd import SGD
from shb import SHB
from nshb import NSHB
from my_models.resnet import resnet18
from my_models.wideresnet import WideResNet28_10
from my_utils import progress_bar

norm_list = []
steps = 0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p_norm = 0
    global norm_list
    global steps
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        steps += 1
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    training_acc = 100.*correct/total
    wandb.log({'training_acc': training_acc,
               'training_loss': train_loss/(batch_idx+1)})

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    wandb.log({'accuracy': acc}) 

def get_full_grad_list(net, train_set, optimizer):
    parameters=[p for p in net.parameters()]
    batch_size=1000
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    device='cuda:0'
    init=True
    full_grad_list=[]

    for i, (xx,yy) in (enumerate(train_loader)):
        xx = xx.to(device, non_blocking = True)
        yy = yy.to(device, non_blocking = True)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
        loss.backward()
        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init=False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(train_set)) * g
    return full_grad_list

def norm_work(norm_list, norm):
    norm_list.append(norm)
    average = sum(norm_list) / len(norm_list)
    return average

class LogitNormalizationWrapper(nn.Module):
    def __init__(self, model, normalize_logits=False):
        super(LogitNormalizationWrapper, self).__init__()
        self.model = model
        self.normalize_logits = normalize_logits

    def forward(self, x):
        out = self.model(x)
        if self.normalize_logits:
            out = out - out.mean(dim=-1, keepdim=True)
            out_norms = out.norm(dim=-1, keepdim=True)
            out_norms = torch.max(out_norms, 10**-10 * torch.ones_like(out_norms))
            out = out / out_norms
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100, cifar10")
    parser.add_argument('--model', default="ResNet18", type=str, help="ResNet18, WideResNet-28-10")
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=64, type=int, help='training batch size')
    parser.add_argument('--repeat', default=0, type=int)
    parser.add_argument('--n_eval_sharpness', default=1024, type=int, help='#examples to evaluate on sharpness')
    parser.add_argument('--bs_sharpness', default=128, type=int, help='batch size for sharpness experiments')
    parser.add_argument('--data_augm_sharpness', action='store_true')   
    parser.add_argument('--algorithm', default='m_apgd_linf', choices=['avg_l2','avg_linf','m_apgd_l2','m_apgd_linf'], type=str)
    parser.add_argument('--rho', default=0.0002, type=float, help='L2 radius for sharpness')
    parser.add_argument('--n_iters', default=20, type=int, help='number of iterations for sharpness')
    parser.add_argument('--n_restarts', default=1, type=int, help='number of restarts for sharpness')
    parser.add_argument('--step_size_mult', default=1.0, type=float, help='step size multiplier for sharpness')
    parser.add_argument('--sharpness_rand_init', action='store_true', help='random initialization')
    parser.add_argument('--no_grad_norm', action='store_true', help='no gradient normalization in APGD')
    parser.add_argument('--normalize_logits', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--sharpness_on_test_set', action='store_true', help='compute sharpness on the test set')
    
    args = parser.parse_args()
    wandb_project_name = "new-sigma-sharpness"
    wandb_exp_name = f"{args.optimizer},b={args.batchsize},lr={args.lr}"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "XXXXXX")
    wandb.init(settings=wandb.Settings(start_method='fork'))

    print('==> Preparing data..')
    if args.dataset == "cifar100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(15),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda:0'
    if args.model == "ResNet18":
        net = resnet18()
    elif args.model == "WideResNet-28-10":
        net = WideResNet28_10()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.0)
    print(optimizer)

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)

    if args.method == "sharpness":
        sharpness_split = 'test' if args.sharpness_on_test_set else 'train'
        loss_f = lambda logits, y: F.cross_entropy(logits, y, reduction='mean')
        net = LogitNormalizationWrapper(net, normalize_logits=args.normalize_logits)
        batches_sharpness = data.get_loaders(args.dataset, args.n_eval_sharpness, args.bs_sharpness, split=sharpness_split, shuffle=False,
                                             data_augm=args.data_augm_sharpness, drop_last=False, randaug=args.data_augm_sharpness)

        if args.algorithm == 'm_apgd_l2':
            sharpness_obj, sharpness_err, _, output = sharpness.eval_APGD_sharpness(
                net, batches_sharpness, loss_f, 
                rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
                rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
                verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='l2')

        elif args.algorithm == 'm_apgd_linf':
            sharpness_obj, sharpness_err, _, output = sharpness.eval_APGD_sharpness(
                net, batches_sharpness, loss_f,
                rho=args.rho, n_iters=args.n_iters, n_restarts=args.n_restarts, step_size_mult=args.step_size_mult,
                rand_init=args.sharpness_rand_init, no_grad_norm=args.no_grad_norm,
                verbose=True, return_output=True, adaptive=args.adaptive, version='default', norm='linf')

        elif args.algorithm == 'avg_l2':
            sharpness_obj, sharpness_err, _, output = sharpness.eval_average_sharpness(
                net, batches_sharpness, loss_f, rho=args.rho, n_iters=args.n_iters, return_output=True, adaptive=args.adaptive, norm='l2')

        elif args.algorithm == 'avg_linf':
            sharpness_obj, sharpness_err, _, output = sharpness.eval_average_sharpness(
                net, batches_sharpness, loss_f, rho=args.rho, n_iters=args.n_iters, return_output=True, adaptive=args.adaptive, norm='linf')

        print('sharpness: obj={:.5f}, err={:.2%}'.format(sharpness_obj, sharpness_err))
        wandb.log({'sharpness_obj': sharpness_obj,
                   'sharpness_err': sharpness_err})
