from __future__ import division
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.cuda as tc
import torch.utils.data.distributed
from configparser import ConfigParser
import sys, os, warnings
import numpy as np
import pickle
from MyImageFolder import ImagesListFileFolder
import gc
import shutil

def get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path):
    import re
    datasets_mean_std_file = open(datasets_mean_std_file_path, 'r').readlines()
    for line in datasets_mean_std_file:
        line = line.strip().split(':')
        dataset_name = line[0]
        dataset_stat = line[1]
        if dataset_name == normalization_dataset_name:
            dataset_stat = dataset_stat.split(';')
            dataset_mean = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[0])]
            dataset_std = [float(e) for e in re.findall(r'\d+\.\d+', dataset_stat[1])]
            return dataset_mean, dataset_std
    print('Invalid normalization dataset name')
    sys.exit(-1)


if len(sys.argv) != 4:
    print('Arguments: config, last, batch in {1, ..., nb_classes//B}')
    sys.exit(-1)

cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)

cp = cp['config']
nb_classes = int(cp['nb_classes'])
normalization_dataset_name = cp['dataset']
first_batch_size = int(cp["first_batch_size"])
P = first_batch_size
batch_size = int(cp["batch_size"])
feat_root = cp["feat_root"]
list_root = cp["list_root"]
model_root = cp["model_root"]
random_seed = int(cp["random_seed"])
num_workers = int(cp['num_workers'])
datasets_mean_std_file_path = cp['mean_std']
dataset_mean, dataset_std = get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
last = sys.argv[2]
nb_batch_to_compute = int(sys.argv[3])
arg_batch = sys.argv[3]
images_list_dir = os.path.join(list_root, normalization_dataset_name)
output_dir = os.path.join(model_root,normalization_dataset_name,"seed"+str(random_seed),"b"+str(first_batch_size))
destination_dir = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size))
destination_dir_last = os.path.join(feat_root, normalization_dataset_name, "seed"+str(random_seed),"b"+str(first_batch_size),last)
print('normalization dataset name = ' + str(normalization_dataset_name))
print('dataset mean = ' + str(dataset_mean))
print('dataset std = ' + str(dataset_std))
gpu = 0
normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
data_types = ['train', 'test']
print("Number of workers = " + str(num_workers))
print("Batch size = " + str(batch_size))
print("Running on gpu " + str(gpu))


if arg_batch=='1':
    model = torch.load(os.path.join(output_dir,f'scratch.pth'))

    features_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    features_extractor.eval()
    features_extractor = features_extractor.cuda(gpu)
    
    for data_type in data_types:
        images_list = os.path.join(images_list_dir,data_type+'.lst')

        print('Loading list file from ' + images_list)

        data_type_destination_dir = os.path.join(destination_dir, 'batch1', data_type)
        if not os.path.exists(os.path.join(data_type_destination_dir,str(nb_classes-1))): 
            try:
                print('cleaning',data_type_destination_dir,'...')
                shutil.rmtree(data_type_destination_dir)
            except:
                pass
            os.makedirs(data_type_destination_dir,exist_ok=True)
            print(data_type_destination_dir,'cleaned!')
            dataset = ImagesListFileFolder(
                images_list, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]),
                    return_path=True,
                    range_classes=None
                    )

            print(data_type + "-set size = " + str(len(dataset)))

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=False)
            
            features_names = {}
            file_names = {}
            last_class = -1
            for data in loader:
                (inputs, labels), _ = data
                inputs = inputs.cuda(gpu)
                features = features_extractor(inputs)
                lablist=list(labels.data.cpu().numpy().squeeze())
                featlist=list(features.data.cpu().numpy().squeeze())
                for i in range(len(lablist)):
                    cu_class = lablist[i]
                    if cu_class!=last_class:
                        last_class=cu_class
                        print('beginning of extraction of class',last_class)
                    with open(os.path.join(data_type_destination_dir,str(cu_class)), 'a') as features_out:
                        features_out.write(str(' '.join([str(e) for e in list(featlist[i])])) + '\n')
        else:
            print(os.path.join(data_type_destination_dir,str(nb_classes-1)), 'exists!')

# subsequent states
else:
    batch_number =int(arg_batch)
    model = torch.load(os.path.join(output_dir,last,f'batch{batch_number}.pth'))

    features_extractor = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    features_extractor.eval()
    features_extractor = features_extractor.cuda(gpu)
    for data_type in data_types:
        images_list = os.path.join(images_list_dir,data_type+'.lst')

        print('Loading list file from ' + images_list)
        data_type_destination_dir = os.path.join(destination_dir_last, 'batch'+str(batch_number), data_type)
        if data_type!='test':
            first_class_batch = (batch_number-1)*P
            last_class_batch = (batch_number*P)-1
        else:
            first_class_batch = 0
            last_class_batch = nb_classes-1
        if not os.path.exists(os.path.join(data_type_destination_dir,str(last_class_batch))): 
            try:
                print('cleaning',data_type_destination_dir,'...')
                shutil.rmtree(data_type_destination_dir)
            except:
                pass
            os.makedirs(data_type_destination_dir,exist_ok=True)
            print(data_type_destination_dir,'cleaned!')
            dataset = ImagesListFileFolder(
                images_list, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize, ]),
                    return_path=True,
                    range_classes=range(first_class_batch,last_class_batch+1)
                    )

            print(data_type + "-set size = " + str(len(dataset)))

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size//2, shuffle=False,
                num_workers=num_workers, pin_memory=False)
            
            features_names = {}
            file_names = {}
            last_class = -1
            for data in loader:
                (inputs, labels), _ = data
                inputs = inputs.cuda(gpu)
                features = features_extractor(inputs)
                lablist=list(labels.data.cpu().numpy().squeeze())
                featlist=list(features.data.cpu().numpy().squeeze())
                for i in range(len(lablist)):
                    cu_class = lablist[i]
                    if cu_class!=last_class:
                        last_class=cu_class
                        print('beginning of extraction of class',last_class)
                    with open(os.path.join(data_type_destination_dir,str(cu_class)), 'a') as features_out:
                        features_out.write(str(' '.join([str(e) for e in list(featlist[i])])) + '\n')
