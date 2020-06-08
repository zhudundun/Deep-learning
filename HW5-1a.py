#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x, t):
    
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)
    
        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)
    
        return self.loss(h[:,0],t), h[:,0] 


# In[4]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io
import argparse


parser = argparse.ArgumentParser(description="1a - BOW Sentiment Analysis")

parser.add_argument("--lr", default=0.001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=6, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=200, type=int, 
                    help="batch size")
parser.add_argument("--vocab_size", default=8000, type=int, 
                    help="dimension of embedded feature")
parser.add_argument("--num_hidden_units", default=500, type=int, 
                    help="dimension of embedded feature")
parser.add_argument("--optimizer", default='adam', const='adam', nargs='?',
                    choices=['adam', 'sgd'],
                    help="dimension of embedded feature")                    



args = parser.parse_args()
print("Hyperparameters:\n", args)

vocab_size = args.vocab_size
num_hidden_units = args.num_hidden_units

LR = args.lr
opt = args.optimizer
batch_size = args.batch_size
no_of_epochs = args.epochs

print("==> Loading data and model...")

x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)

x_train = x_train[0:25000]

y_train = np.zeros((25000,))
y_train[0:12500] = 1


x_test = []
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)

y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = BOW_model(vocab_size, num_hidden_units)
model.cuda()

if opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif opt == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []

print("==> Start training...")

for epoch in range(no_of_epochs):

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input,target)
        loss.backward()

        optimizer.step()   
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(x_input,target)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

print("==> Saving model...")

torch.save(model,'BOW.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data.npy',data)


# In[ ]:




