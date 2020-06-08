


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.distributed as dist



import os
import subprocess
from mpi4py import MPI

import torchvision
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import sys
import random



import argparse

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor






def data_loader_and_transformer(root_path):
   
    train_data_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_data_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
   
    # Data loader.
    train_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=True,
        download=True,
        transform=train_data_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=False,
        download=True,
        transform=test_data_tranform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0
    )

    return train_data_loader, test_data_loader








class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=start_stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Project the x into correct dimension if needed
        self.downsample = downsample
        if self.downsample:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=start_stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels)                
            )
    
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.downsample:
            residual = self.projection(residual)
        x += residual

        return x







class ResNet(nn.Module):
    def __init__(self, num_blocks_list, num_classes=100):
        super(ResNet, self).__init__()
        self.curt_in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32) 
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.conv2_x = self._add_layers(32, num_blocks_list[0])
        self.conv3_x = self._add_layers(64, num_blocks_list[1], start_stride=2)
        self.conv4_x = self._add_layers(128, num_blocks_list[2], start_stride=2)
        self.conv5_x = self._add_layers(256, num_blocks_list[3], start_stride=2)
        
        self.maxpool = nn.MaxPool2d(4, stride=1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _add_layers(self, out_channels, num_blocks, start_stride=1):
        downsample = False
        if start_stride != 1 or self.curt_in_channels != out_channels:
            downsample = True
        
        layers = []
        layers.append(BasicBlock(self.curt_in_channels, out_channels,
                                 start_stride=start_stride, downsample=downsample))
        self.curt_in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.curt_in_channels, out_channels))
        
        return nn.Sequential(*layers)





def test(
    net, 
    criterion, 
    test_data_loader, 
    
    debug=False
    ):
    
    net.eval()

    
    running_loss = 0
    total_correct = 0
    total_samples = 0

    
    for batch_index, (images, labels) in enumerate(test_data_loader):
       
        if debug and total_samples >= 10001:
            return
            
        images = Variable(images,volatile=True).cuda()
        labels = Variable(labels,volatile=True).cuda()

        outputs = net(images)
        loss = criterion(outputs, labels.long())
            
        running_loss += loss.data[0]
        curt_loss = running_loss / (batch_index + 1)

        _, predict_label = torch.max(outputs, 1)
        total_samples += labels.shape[0]
        total_correct += predict_label.eq(labels.long()).float().sum().data[0]
        accuracy = total_correct / total_samples

       
    
    print("Testing [finished] accuracy: %.5f" % accuracy)








def train(
    net, 
    criterion, 
    optimizer,
    
    
    epochs,
    train_data_loader,
    test_data_loader,
    
    lr_schedule=False,
    debug=False
    ):
    
    for curt_epoch in range(1, epochs):
        net.train()

        if curt_epoch > 10:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000

       
        running_loss = 0
        total_correct = 0
        total_samples = 0

        if lr_schedule:
            scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
            scheduler.step()

        for batch_index, (images, labels) in enumerate(train_data_loader):
           
            if debug and total_samples >= 10001:
                return
            
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = net(images)

          
            loss = criterion(outputs, labels.long())
            loss.backward()
            
            ###########
            for param in net.parameters():
                tensor0 = param.grad.data.cpu()
                dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
                tensor0 /= float(num_nodes)
                param.grad.data = tensor0.cuda()        
        

            optimizer.step()

            running_loss += loss.data[0]
            curt_loss = running_loss / (batch_index + 1)

            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().data[0]
            accuracy = total_correct / total_samples
            
            
            
        
        print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (curt_epoch + 1, curt_loss, accuracy))
        
        test(net, criterion, test_data_loader,  debug=debug)
    
    print("Training [finished]")














def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Training ResNet on CIFAR100")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--epochs", default=40, type=int, help="number of training epochs")
parser.add_argument("--lr_schedule", default=True, type=str2bool, help="perform lr shceduling")
parser.add_argument("--show_sample_image", default=False, type=str2bool, help="display data insights")
parser.add_argument("--debug", default=False, type=str2bool, help="using debug mode")
parser.add_argument("--data_path", default="./data", type=str, help="path to store data")
args = parser.parse_args()


def main():
    
    torch.manual_seed(72)
    torch.cuda.manual_seed(72)
    np.random.seed(72)
    random.seed(72)

    print("*** Performing data augmentation...")
    train_data_loader, test_data_loader = data_loader_and_transformer(
                                                args.data_path)

    if args.show_sample_image:
        print("*** Loading image sample from a batch...")
        data_iter = iter(train_data_loader)
        images, labels = data_iter.next() 
        print("images type {}, shape {}".format(images.type(), images.shape))
        print("shape of a single image", images[0].shape)
        print("labels type {}, shape {}".format(labels.type(), labels.shape))
        print("label for the first 4 images", labels[:4])
        
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")


   
    print("*** Initializing model...")
    resnet = ResNet([2, 4, 4, 2])
        
    #########
    for param in resnet.parameters():
        tensor0 = param.data
        dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
        param.data = tensor0/np.sqrt(np.float(num_nodes))
    
    resnet.cuda()
    resnet = torch.nn.DataParallel(resnet)
    cudnn.benchmark = True
    
   

    
    print("* Hyperparameters: LR = {}, EPOCHS = {}, LR_SCHEDULE = {}"
          .format(args.lr, args.epochs, args.lr_schedule))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr)
    train(
        resnet,
        criterion,
        optimizer,
       
        
        args.epochs,
        train_data_loader,
        test_data_loader,
       
        lr_schedule=args.lr_schedule,
        debug=args.debug
    )

   
    print("*** Start testing...")
    test(
        resnet,
        criterion,
        test_data_loader,
        
        debug=args.debug
    )
    


main()
    

