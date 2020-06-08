#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import model_zoo
import os

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random



import argparse


def data_loader(root_path):
   

    train_data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_data_tranform = transforms.Compose([
        transforms.Resize(224),
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=False,
        download=True,
        transform=test_data_tranform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader






model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=True) :
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir = './pretrained'))
    return model





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
    def __init__(self, blocks_list, num_classes=100):
        super(ResNet, self).__init__()
        self.curr_in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32) 
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self.conv2_x = self.adding_layer(32, blocks_list[0])
        self.conv3_x = self.adding_layer(64, blocks_list[1], start_stride=2)
        self.conv4_x = self.adding_layer(128, blocks_list[2], start_stride=2)
        self.conv5_x = self.adding_layer(256, blocks_list[3], start_stride=2)
        
        self.max_pool = nn.MaxPool2d(4, stride=1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def adding_layer(self, out_channels, num_blocks, start_stride=1):
        downsample = False
        if start_stride != 1 or self.curr_in_channels != out_channels:
            downsample = True
        
        layers = []
        layers.append(BasicBlock(self.curr_in_channels, out_channels,
                                 start_stride=start_stride, downsample=downsample))
        self.curr_in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.curr_in_channels, out_channels))
        
        return nn.Sequential(*layers)






def test(
    net, 
    criterion, 
    test_loader, 
    device
    ):
   

    net.eval()

    running_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
          
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels.long())
            
            running_loss += loss.item()
            curr_loss = running_loss / (batch_index + 1)

            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

          
    
    print("Testing [finished] accuracy: %.5f" % accuracy)








def train(
    net, 
    criterion, 
    optimizer,
    
    epochs,
    train_loader,
    test_loader,
    device,
    lr_schedule=False
    
    ):
   

    for curr_epoch in range(1, epochs):
        net.train()

        if curr_epoch > 10:
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

        for batch_index, (images, labels) in enumerate(train_loader):
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels.long())
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            curr_loss = running_loss / (batch_index + 1)

            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
            
        
        
        print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (curr_epoch + 1, curr_loss, accuracy))
        
        test(net, criterion, test_loader, device)
    
    print("Training [finished]")













def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Training ResNet on CIFAR100")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--epochs", default=40, type=int, help="number of training epochs")
parser.add_argument("--lr_schedule", default=True, type=str2bool, help="perform lr shceduling")
parser.add_argument("--show_sample_image", default=False, type=str2bool, help="display data insights")
parser.add_argument("--data_path", default="~/scratch/", type=str, help="path to store data")
args = parser.parse_args()


def main():
   
    torch.manual_seed(72)
    torch.cuda.manual_seed(72)
    np.random.seed(72)
    random.seed(72)

   
    print("*** Performing data augmentation...")
    train_loader, test_loader = data_loader(
                                                args.data_path)

    
    if args.show_sample_image:
        print("*** Loading image sample from a batch...")
        data_iter = iter(train_loader)
        images, labels = data_iter.next()  # Retrieve a batch of data
        
       
        print("images type {}, shape {}".format(images.type(), images.shape))
       
        print("shape of a single image", images[0].shape)
       
        print("labels type {}, shape {}".format(labels.type(), labels.shape))
        
        print("label for the first 4 images", labels[:4])
        
        
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

   
   
    print("*** Initializing pre-trained model...")
    resnet = resnet18()
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, 100)


    resnet = resnet.to(device)
    if device == 'cuda':
        resnet = torch.nn.DataParallel(resnet)
        cudnn.benchmark = True
    
   

   
    print("*** Start training on device {}...".format(device))
    print("* Hyperparameters: LR = {}, EPOCHS = {}, LR_SCHEDULE = {}"
          .format(args.lr, args.epochs, args.lr_schedule))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr)
    train(
        resnet,
        criterion,
        optimizer,
        
        args.epochs,
        train_loader,
        test_loader,
        device,
        lr_schedule=args.lr_schedule
       
    )

    
    print("*** Start testing...")
    test(
        resnet,
        criterion,
        test_loader,
        device
        
    )
    
main()

