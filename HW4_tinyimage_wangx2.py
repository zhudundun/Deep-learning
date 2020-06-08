


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import sys
import numpy as np
import random



import argparse
def create_val_folder(val_dir):
    
    path = os.path.join(val_dir, 'images')
   
    filename = os.path.join(val_dir, 'val_annotations.txt')
    fp = open(filename, "r") # open file in read mode
    data = fp.readlines() # read line by line
    '''
    Create a dictionary with image names as key and
    corresponding classes as values
    '''
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath): # check if folder exists
            os.makedirs(newpath)
    # Check if image exists in default directory
        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return




def data_loader_and_transformer():

    train_dir = '/u/training/tra352/scratch/tiny-imagenet-200/train'
    train_dataset = torchvision.datasets.ImageFolder(train_dir,
    transform=transforms.Compose([
           transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
                ]))
    print(train_dataset.class_to_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=128, shuffle=True, num_workers=8)
    val_dir = '/u/training/tra352/scratch/tiny-imagenet-200/val/'
    if 'val_' in os.listdir(val_dir+'images/')[0]:
        create_val_folder(val_dir)
        val_dir = val_dir+'images/'
    else:
        val_dir = val_dir+'images/'

    
    val_dataset = torchvision.datasets.ImageFolder(val_dir,
    transform=transforms.Compose([

               transforms.ToTensor() ]))
    print(val_dataset.class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=100, shuffle=False, num_workers=8)
    return train_loader, val_loader
   





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
    def __init__(self, num_blocks_list, num_classes=200):
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
        
        self.maxpool = nn.MaxPool2d(4, stride=2)
        self.fc = nn.Linear(256*3*3, num_classes)
    
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
    device,
    debug=False
    ):
   
    net.eval()
    running_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        

        for batch_index, (images, labels) in enumerate(test_data_loader):
            if debug and total_samples >= 10001:
                return
            
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

         
    print("Testing [finished] accuracy: %.5f" % accuracy)








def train(
    net, 
    criterion, 
    optimizer,
    
    epochs,
    train_data_loader,
    test_data_loader,
    device,
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
            
         
            
        
        print('Training [epoch: %d] loss: %.3f, accuracy: %.5f' %
                (curt_epoch + 1, curt_loss, accuracy))
        
        test(net, criterion, test_data_loader, device, debug=debug)
    
    print("Training [finished]")










def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Training ResNet on CIFAR100")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--epochs", default=40, type=int, help="number of training epochs")
parser.add_argument("--lr_schedule", default=True, type=str2bool, help="perform lr shceduling")
parser.add_argument("--show_sample_image", default=False, type=str2bool, help="display data insights")
parser.add_argument("--debug", default=False, type=str2bool, help="using debug mode")
parser.add_argument("--data_path", default="/u/training/tra352/scratch/tiny-imagenet-200/train", type=str, help="path to store data")
args = parser.parse_args()


def main():
   
    torch.manual_seed(72)
    torch.cuda.manual_seed(72)
    np.random.seed(72)
    random.seed(72)

    print("*** Performing data augmentation...")
    train_data_loader, test_data_loader = data_loader_and_transformer()

    if args.show_sample_image:
        print("*** Loading image sample from a batch...")
        data_iter = iter(train_data_loader)
        images, labels = data_iter.next()  # Retrieve a batch of data
        
       
        print("images type {}, shape {}".format(images.type(), images.shape))
        print("shape of a single image", images[0].shape)
        print("labels type {}, shape {}".format(labels.type(), labels.shape))
        print("label for the first 4 images", labels[:4])
        
        plt.imshow(images[0][0].numpy())
        plt.savefig("sample_image.png")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    print("*** Initializing model...")
    resnet = ResNet([2, 4, 4, 2])

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
        train_data_loader,
        test_data_loader,
        device,
        lr_schedule=args.lr_schedule,
        debug=args.debug
    )

    print("*** Start testing...")
    test(
        resnet,
        criterion,
        test_data_loader,
        device,
        debug=args.debug
    )
    

main()












