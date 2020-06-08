#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class StatefulLSTM(nn.Module):
    def __init__(self,in_size,out_size):
        super(StatefulLSTM,self).__init__()
        
        self.lstm = nn.LSTMCell(in_size,out_size)
        self.out_size = out_size
        
        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self,x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x,(self.h,self.c))

        return self.h

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x

class RNN_model(nn.Module):
    def __init__(self,no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.lstm1 = StatefulLSTM(300,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() 

  

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
       

    def forward(self, x, t, train=True):

        no_of_timesteps = x.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(x[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.5,train=train)

           

            outputs.append(h)

        outputs = torch.stack(outputs) 
        outputs = outputs.permute(1,2,0) 

        pool = nn.MaxPool1d(x.shape[1])
        h = pool(outputs)
        h = h.view(h.size(0),-1)

        h = self.fc_output(h)

        return self.loss(h[:,0],t), h[:,0]


# In[ ]:


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

from RNN_model import RNN_model

parser = argparse.ArgumentParser(description="2b - RNN Sentiment Analysis with Glove Embedding")

parser.add_argument("--lr", default=0.001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=20, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=200, type=int, 
                    help="batch size")
parser.add_argument("--vocab_size", default=100000, type=int, 
                    help="dimension of embedded feature")
parser.add_argument("--num_hidden_units", default=500, type=int, 
                    help="dimension of embedded feature")
parser.add_argument("--optimizer", default='adam', const='adam', nargs='?',
                    choices=['adam', 'sgd'],
                    help="dimension of embedded feature")
parser.add_argument("--seq_len_train", default=100, type=int,
                    help="sequence length for training")                
parser.add_argument("--seq_len_test", default=100, type=int,
                    help="sequence length for testing")

args = parser.parse_args()
print("Hyperparameters:\n", args)


vocab_size = args.vocab_size
num_hidden_units = args.num_hidden_units   

LR = args.lr
opt = args.optimizer
batch_size = args.batch_size
no_of_epochs = args.epochs

sequence_lengths = [args.seq_len_train, args.seq_len_test]

print("==> Loading data and model...")

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
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

model = RNN_model(num_hidden_units)
model.cuda()

if opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif opt=='sgd' :
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

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = sequence_lengths[0]
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
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

    print("epoch, training accuracy, training loss, eplased time")
    print(epoch, 
          "%.2f" % (epoch_acc * 100.0), 
          "%.4f" % epoch_loss, 
          "%.4f" % float(time.time()-time1))

    if (epoch + 1) % 5 == 0:
        print("=> Saving model...")
        torch.save(model, 'rnn.model')

    if (epoch + 1) % 3 == 0:

        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()
        
        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):

            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            sequence_length = sequence_lengths[1]
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            x_input = glove_embeddings[x_input]
            y_input = y_test[I_permutation[i:i+batch_size]]

            data = Variable(torch.FloatTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()

            with torch.no_grad():
                loss, pred = model(data, target, train=False)
            
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

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss,
              "%.4f" % float(time_elapsed))

print("==> Saving model...")

torch.save(model, 'rnn.model')
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('rnn_data.npy',data)


# In[ ]:


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

from RNN_model import RNN_model

parser = argparse.ArgumentParser(description="2b - RNN Sentiment Analysis Test")

parser.add_argument("--epochs", default=10, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=200, type=int, 
                    help="batch size")
parser.add_argument("--vocab_size", default=8000, type=int, 
                    help="dimension of embedded feature")

args = parser.parse_args()
print("Hyperparameters:\n", args)



batch_size = args.batch_size
no_of_epochs = args.epochs
vocab_size = args.vocab_size


print("==> Loading data and model...")

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
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

model = torch.load('rnn.model')
model.cuda()

L_Y_test = len(y_test)

test_accu = []

print("==> Start testing...")

for epoch in range(no_of_epochs):


    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]

       
        sequence_length = (epoch + 1) * 50

        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data, target, train=False)
        
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

    print("sequence length, test accuracy, test loss, eplased time")
    print(sequence_length, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss,
          "%.4f" % float(time_elapsed))

