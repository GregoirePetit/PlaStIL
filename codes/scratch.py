#!/usr/bin/env python
# coding=utf-8
import sys
import os
import time
import shutil
import socket
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from datetime import timedelta
from configparser import ConfigParser

from AverageMeter import AverageMeter
from MyImageFolder import ImagesListFileFolder
from Utils import DataUtils

if len(sys.argv) != 2:
    print('Arguments: config')
    sys.exit(-1)

cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)

cp = cp['config']
nb_classes = int(cp['nb_classes'])
normalization_dataset_name = cp['dataset']
first_batch_size = int(cp["first_batch_size"])
feat_root = cp["feat_root"]
list_root = cp["list_root"]
model_root = cp["model_root"]
random_seed = int(cp["random_seed"])
num_workers = int(cp['num_workers'])
epochs = int(cp['epochs'])

batch_size = int(cp['batch_size'])
lr = float(cp['lr'])
momentum = float(cp['momentum'])
weight_decay = float(cp['weight_decay'])
lrd = int(cp['lrd'])


B = first_batch_size
datasets_mean_std_file_path = cp["mean_std"]
output_dir = os.path.join(model_root,normalization_dataset_name,"seed"+str(random_seed),"b"+str(first_batch_size))
train_file_path = os.path.join(list_root,normalization_dataset_name,"train.lst")
test_file_path = os.path.join(list_root,normalization_dataset_name,"test.lst")

utils = DataUtils()
train_batch_size       = 128
test_batch_size        = 50
eval_batch_size        = 128
base_lr                = 0.1
lr_strat               = [30, 60]
lr_factor              = 0.1
custom_weight_decay    = 0.0001
custom_momentum        = 0.9

print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '\n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
top = min(5, B)



if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if device is not None:
    print("Use GPU: {} for training".format(device))

# instantiate a ResNet18 model
model = models.resnet18()

model.fc = nn.Linear(512, B)


model.cuda(device)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

print('modele charge')

val_dataset = ImagesListFileFolder(
            test_file_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), random_seed=random_seed, range_classes=range(B), nb_classes=nb_classes)

train_dataset = ImagesListFileFolder(
            train_file_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AugMix(severity=5,chain_depth=7),
                transforms.ToTensor(),
                normalize,
            ]), random_seed=random_seed, range_classes=range(B), nb_classes=nb_classes)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=False)

print('Classes number = {}'.format(len(train_dataset.classes)))
print('Training dataset size = {}'.format(len(train_dataset)))
print('Validation dataset size = {}'.format(len(val_dataset)))

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch==80 or epoch==120 or epoch==160:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

print('\nstarting training...')
start = time.time()
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, lr)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, (input, target) in enumerate(train_loader):
        if device is not None:
            input = input.cuda(device)
        target = target.cuda(device)
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    epoch_time =  timedelta(seconds=round(end - start))

    print('{:03}/{:03} | Train ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f} | loss = {:.4f}'.format(
        epoch+1, epochs,  len(train_loader), top1.avg, top, top5.avg, losses.avg))
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    print('        | Test  ({})'.format(len(val_loader))+' '*(len(str(len(train_loader)))-len(str(len(val_loader))))+' |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(top1.avg, top, top5.avg))
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

ckp_name = os.path.join(output_dir,'scratch.pth')
torch.save(model, ckp_name)
