import sys
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import time
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp.grad_scaler import GradScaler
import torchvision
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from argparse import ArgumentParser
import torchvision.transforms as transforms
import resnet

import json
import random

def seed_everything(seed=42):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # print(torch.rand(16).dtype, np.random.rand(16).dtype)
    torch.backends.cudnn.deterministic = False#True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def train(args):

        # args = q.get()
        # rank = gpu #args.nr * args.gpus + gpu
        #print(rank)
        # ngpus = args.gpus	                          
        # dist.init_process_group(                                   
        # backend='nccl',                                         
        #         init_method='env://',                                   
        # world_size=args.world_size,                              
        # rank=rank                                               
        # )                                                          
        # seed_everything(42)
        # torch.cuda.set_device(gpu)

        transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train = torchvision.datasets.CIFAR10(root=args.path_to_dataset, train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=2)

        test = torchvision.datasets.CIFAR10(root=args.path_to_dataset, train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(test, batch_size=256,shuffle=False, num_workers=2)

        print(len(trainloader) , len(testloader))

        # log_file = open(args.output + '/log.txt', "a")

        model = resnet.ResNet50ModelParallel( gpus = (0 , 1) )

        # model.cuda(gpu)

        count = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1 * args.gpus, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

        # model = nn.parallel.DistributedDataParallel(model,
        #                                         device_ids=[gpu], find_unused_parameters=True)


        EPOCHS = 200
        for epoch in range(EPOCHS):
                losses = []
                running_loss = 0

                start = time.time()
                
                for i, inp in enumerate(trainloader):
                        inputs, labels = inp
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        optimizer.zero_grad()
                
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        losses.append(loss.item())

                        loss.backward()
                        optimizer.step()

                        if args.profile:
                                profiler.step()
                        
                        running_loss += loss.item()
                        
                        if i%5== 0 and i > 0:
                                end = time.time()
                                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100 , " throuput : ", ( 5 * args.batchsize ) / (end - start) , " imgs/sec " )
                                start = time.time()

                        running_loss = 0.0

                avg_loss = sum(losses)/len(losses)
                scheduler.step(avg_loss)
            
        print('Training Done')
            

#     torch.save(model.state_dict(), args.output + f'/final_model.pth')
#     log_file.close()




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument('-ptd', '--path_to_dataset', default='~/cifar', type=str,
                        help='output model directory')
    parser.add_argument('-o', '--output', default='/glb/home/inatjv/Builds/CT_Image_Recon_Dev/bin/training_output/siamese/', type=str,
                        help='output model directory')
    parser.add_argument('-l', '--loss', default='weighted_BCE', type=str,
                        help='loss function type: weighted_BCE, Focal_loss, TCE, RecallCE, Lovasz, OhemCE, SoftIOU0, SoftIOU1')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-cl', '--n_class', default=3, type=int,
                        help='number of classes in labels: 2 or 3')
    parser.add_argument('-bs', '--batchsize', default=256, type=int,
                        help='batch size per gpu. default=256')
    parser.add_argument('-profile', '--profile', default=False, type=bool,
                        help='whether to profile the code')
    parser.add_argument('-lr', '--lr', default=0.0001, type=float,
                        help='learning rate')

    args = parser.parse_args() 

    args.world_size = args.gpus * args.nodes     

    args.batch_number = 200           

#     print(args.output)

    p = train( args )

