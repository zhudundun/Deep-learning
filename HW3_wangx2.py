import os


import torch
import torchvision
import torchvision.transforms as transforms


def data_loader_and_transformer(root_path):
   
    train_data_transform = transforms.Compose([
        transforms.RandomCrop(size=[32,32], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_data_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=root_path,
        train=True,
        download=True,
        transform=train_data_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=root_path,
        train=False,
        download=True,
        transform=test_data_tranform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    return train_data_loader, test_data_loader




import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        

        super(DeepCNN, self).__init__()
        self.cnov = self._add_conv_layers()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        

        x = self.cnov(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _add_conv_layers(self):
        out_channels_list = [
            64, 64, 'pool',
           
            64, 64, 'pool',
            64, 64, 64, 'pool',
            512, 512, 512, 'pool',
            512, 512, 512, 'pool'
        ]

        layers = []
        in_channels = 3

        for out_channels in out_channels_list:
            if out_channels == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3,
                    stride=1,
                    padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

                

        return nn.Sequential(*layers)




import torch
import torch.nn as nn
import torch.optim as optim


def train(
    net, 
    criterion, 
    optimizer,
    epochs,
    train_data_loader, 
    device,
    lr_schedule=False
    ):
    
    for curt_epoch in range(0, epochs):
        net.train()

        running_loss = 0
        total_correct = 0
        total_samples = 0

        if lr_schedule:
            optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)

        for batch_index, (images, labels) in enumerate(train_data_loader):
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels.long())
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
            
            print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                    (curt_epoch + 1, batch_index + 1, curt_loss, accuracy))
            
 
    print("Training [finish]")




import torch

def test(
    net, 
    criterion, 
    test_data_loader, 
    device,
    ):
   

    net.eval()

    running_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_data_loader):
                      
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels.long())
            
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
                    (batch_index + 1, curt_loss, accuracy))
    
    print("Testing [finish] finial accuracy: %.5f" % accuracy)




import torch
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random




DATA_PATH = "./data"


LR = 0.001
EPOCHS = 15





def main():
   
    torch.manual_seed(72)
    torch.cuda.manual_seed(722)
    np.random.seed(72)
    random.seed(72)

    print("*** Performing data augmentation...")
    train_data_loader, test_data_loader = data_loader_and_transformer(DATA_PATH)


    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("*** Initializing model...")
    cnn = DeepCNN()
    cnn = cnn.to(device)
    if device == 'cuda':
        cnn = torch.nn.DataParallel(cnn)
        cudnn.benchmark = True
    
 
    print("*** Start training on device {}...".format(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    train(
        cnn,
        criterion,
        optimizer,
       
        EPOCHS,
        train_data_loader,
        device,
        lr_schedule=False
       
    )

    print("*** Start testing...")
    test(
        cnn,
        criterion,
        test_data_loader,
        device,
    )
    

main()
