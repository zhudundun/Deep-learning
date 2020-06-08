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
from RNN_language_model import RNN_language_model

parser = argparse.ArgumentParser(description="2a - RNN Sentiment Analysis")

parser.add_argument("--lr", default=0.001, type=float, 
                    help="learning rate")
parser.add_argument("--epochs", default=30, type=int, 
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


x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)


#x_train = x_train[0:25000]


#y_train = np.zeros((25000,))
#y_train[0:12500] = 1


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


model = RNN_model(vocab_size, num_hidden_units)
language_model = torch.load('language.model')


model.embedding.load_state_dict(language_model.embedding.state_dict())
model.lstm1.lstm.load_state_dict(language_model.lstm1.lstm.state_dict())
model.bn_lstm1.load_state_dict(language_model.bn_lstm1.state_dict())
model.lstm2.lstm.load_state_dict(language_model.lstm2.lstm.state_dict())
model.bn_lstm2.load_state_dict(language_model.bn_lstm2.state_dict())
model.lstm3.lstm.load_state_dict(language_model.lstm3.lstm.state_dict())
model.bn_lstm3.load_state_dict(language_model.bn_lstm3.state_dict())
model.cuda()

params = []

for param in model.lstm3.parameters():
    params.append(param)
for param in model.bn_lstm3.parameters():
    params.append(param)
for param in model.fc_output.parameters():
    params.append(param)

if opt == 'adam':
    optimizer = optim.Adam(params, lr=LR)
elif opt=='sgd' :
    optimizer = optim.SGD(params, lr=LR, momentum=0.9)

#L_Y_train = len(y_train)
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
    
    I_permutation = np.random.permutation(len(x_train))

    for i in range(0, len(x_train), batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]

        sequence_length = sequence_lengths[0]

        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data, target, train=True)
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

            x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < sequence_length):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            y_input = y_train[I_permutation[i:i+batch_size]]

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

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss,
              "%.4f" % float(time_elapsed))

print("==> Saving model...")

torch.save(model, 'rnn.model')
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('rnn_data.npy',data)

