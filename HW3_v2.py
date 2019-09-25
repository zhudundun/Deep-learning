#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import torch

# def load_checkpoint(net):
#     print("Loading model from disk...")

#     if not os.path.isdir('checkpoints'):
#         print("Error: no checkpoints available.")
#         raise AssertionError()
    
#     checkpoint = torch.load('checkpoints/model_state.pt')
#     net.load_state_dict(checkpoint['model_state_dict'])
#     start_epoch = checkpoint['epoch']
#     best_acc = checkpoint['best_acc']

#     return start_epoch, best_acc


# def save_checkpoint(net, epoch, best_acc):
#     print("Saving model to disk...")

#     state = {
#         'model_state_dict': net.state_dict(),
#         'epoch': epoch,
#         'best_acc': best_acc
#     }
#     if not os.path.isdir('checkpoints'):
#         os.mkdir('checkpoints')
    
#     torch.save(state, 'checkpoints/model_state.pt')


# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms


def data_loader_and_transformer(root_path):
    """Utils for loading and preprocessing data
    Args:
        root_path(string): the path to download/fetch the data
    Returns:
        train_data_loader(iterator)
        test_data_loader(iterator) 
    """

    # Data augmentation.
    # See https://github.com/kuangliu/pytorch-cifar/issues/19 for the normalization.
    train_data_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_data_tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    # Data loader.
    train_dataset = torchvision.datasets.CIFAR10(
        root=root_path,
        train=True,
        download=True,
        transform=train_data_transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=166,
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
        batch_size=166,
        shuffle=False,
        num_workers=2
    )

    return train_data_loader, test_data_loader


# In[2]:


import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        """Deep CNN model based on VGG16
        See the original paper: https://arxiv.org/abs/1409.1556
        """

        super(DeepCNN, self).__init__()
        self.cnov = self._add_conv_layers()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        """Forward step which will be called directly
        by PyTorch
        """

        x = self.cnov(x)

        # Reshape tensor to match the fc dimensions.
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _add_conv_layers(self):
        # Network structures for CONV block and POOL block.
        out_channels_list = [
            64, 64, 'pool',
           
            64, 64, 'pool',
            64, 64, 64, 'pool',
            512, 512, 512, 'pool',
            512, 512, 512, 'pool'
        ]

        layers = []
        in_channels = 3

        # Build.
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


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim


def train(
    net, 
    criterion, 
    optimizer,
   # best_acc, 
    #start_epoch, 
    epochs,
    train_data_loader, 
    device,
    lr_schedule=False
    #debug=False
    ):
    """Training setup for a single epoch
    Args:
        net(class.DeepCNN)
        criterion(torch.nn.CrossEntropyLoss)
        optimizer(torch.optim.Adam)
        best_acc(float): best accuracy if loaded from checkpoints
        start_epoch(int): start epoch from last checkpoint
        epochs(int): total number of epochs
        train_data_loader(iterator)
        device(str): 'cpu' or 'cuda'
        lr_schedule(bool): whether to perform leanring rate scheduling
        debug(bool): whether to use a debug mode
    """

    #for curt_epoch in range(start_epoch, epochs):
    for curt_epoch in range(0, epochs):
        # Set to train mode.
        net.train()

        # To monitor the training process.
        running_loss = 0
        total_correct = 0
        total_samples = 0

        # Schedule learning rate if specified.
        if lr_schedule:
            optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.6)

        # Traning step.
        for batch_index, (images, labels) in enumerate(train_data_loader):
            # Only train on a smaller subset if debug mode is true
#             if debug and total_samples >= 10001:
#                 return
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            # CrossEntropyLoss wants inputs as torch.FloatTensor, and 
            # targets as torch.LongTensor which are class indices, not float values.
            loss = criterion(outputs, labels.long())
            loss.backward()

            optimizer.step()

            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy.
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples
            
            print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                    (curt_epoch + 1, batch_index + 1, curt_loss, accuracy))
            
            # Update best accuracy and save checkpoint
#             if accuracy > best_acc:
#                 best_acc = accuracy
                #save_checkpoint(net, curt_epoch, best_acc)
    
    print("Training [finished]")


# In[4]:


import torch

def test(
    net, 
    criterion, 
    test_data_loader, 
    device,
    #debug=False
    ):
    """Testing setup for a single epoch
    Args:
        net(class.DeepCNN)
        criterion(torch.nn.CrossEntropyLoss)
        test_data_loader(iterator)
        device(str): 'cpu' or 'cuda'
        debug(bool): whether to use a debug mode
    """

    # Set to test mode.
    net.eval()

    # To monitor the testing process.
    running_loss = 0
    total_correct = 0
    total_samples = 0

    # Testing step.
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_data_loader):
            # Only test on a smaller subset if debug mode is true
#             if debug and total_samples >= 10001:
#                 return
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels.long())
            
            # Loss.
            running_loss += loss.item()
            curt_loss = running_loss / (batch_index + 1)

            # Accuracy
            _, predict_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            total_correct += predict_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
                    (batch_index + 1, curt_loss, accuracy))
    
    print("Testing [finished] finial accuracy: %.5f" % accuracy)


# In[5]:


import torch
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random


# Set to True if you have checkpoints available and want to resume from it
#LOAD_CHECKPOINT = False

# Set to True to get some insights of the data
#SHOW_SAMPLE_IMAGE = False

# Set to True to run in a debug mode which uses less data
#DEBUG = False

DATA_PATH = "./data"

#Hyperparameters.
trials = [
    [0.01, 50],
    [0.001, 50],
    [0.01, 100],
    [0.001, 100],
    [0.001, 70],
    [0.001, 15]]

if (len(sys.argv) == 2):
    trial_number = int(sys.argv[1])
else:
    trial_number = 5
LR = trials[trial_number][0]
EPOCHS = trials[trial_number][1]



def main():
    """High level pipelines.
    Usage: run "python3 main.py trial_num"
    such as "python3 main.py 1"
    """

    # Set seed.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load data.
    print("*** Performing data augmentation...")
    train_data_loader, test_data_loader = data_loader_and_transformer(DATA_PATH)

    # Load sample image.
#     if SHOW_SAMPLE_IMAGE:
#         print("*** Loading image sample from a batch...")
#         data_iter = iter(train_data_loader)
#         images, labels = data_iter.next()  # Retrieve a batch of data
        
#         # Some insights of the data.
#         # images type torch.FloatTensor, shape torch.Size([128, 3, 32, 32])
#         print("images type {}, shape {}".format(images.type(), images.shape))
#         # shape of a single image torch.Size([3, 32, 32])
#         print("shape of a single image", images[0].shape)
#         # labels type torch.LongTensor, shape torch.Size([128])
#         print("labels type {}, shape {}".format(labels.type(), labels.shape))
#         # label for the first 4 images tensor([2, 3, 4, 2])
#         print("label for the first 4 images", labels[:4])
        
#         # Get a sampled image.
#         plt.imshow(images[0][0].numpy())
#         plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model.
    print("*** Initializing model...")
    cnn = DeepCNN()
    # print(cnn)
    cnn = cnn.to(device)
    if device == 'cuda':
        cnn = torch.nn.DataParallel(cnn)
        cudnn.benchmark = True
    
    # Load checkpoint.
    #start_epoch = 0
    #best_acc = 0
#     if LOAD_CHECKPOINT:
#         print("*** Loading checkpoint...")
#         start_epoch, best_acc = load_checkpoint(cnn)

    # Training.
    print("*** Start training on device {}...".format(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    train(
        cnn,
        criterion,
        optimizer,
        #best_acc,
        #start_epoch,
        EPOCHS,
        train_data_loader,
        device,
        lr_schedule=False
        #debug=DEBUG
    )

    # Testing.
    print("*** Start testing...")
    test(
        cnn,
        criterion,
        test_data_loader,
        device,
        #debug=DEBUG
    )
    
    print("*** Congratulations! You've got an amazing model now :)")

if __name__=="__main__":
    main()


# In[ ]:





# In[ ]:




