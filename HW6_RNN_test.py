#!/usr/bin/env python
# coding: utf-8

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

parser = argparse.ArgumentParser(description="2a - RNN Sentiment Analysis")

parser.add_argument("--epochs", default=10, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch_size", default=200, type=int, 
                    help="batch size")

parser.add_argument("--vocab_size", default=8000, type=int, 
                    help="dimension of embedded feature")

args = parser.parse_args()
print("Hyperparameters:\n", args)


vocab_size = args.vocab_size

batch_size = args.batch_size
no_of_epochs = args.epochs

print("==> Loading data and model...")


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

        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
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

