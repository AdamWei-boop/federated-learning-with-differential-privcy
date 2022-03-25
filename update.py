#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from    torch.nn import functional as F
import numpy as np
import random
from opacus import PrivacyEngine

import warnings

warnings.filterwarnings("ignore")
#matplotlib.use('Agg')

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, act):
        self.args = args
        self.act = act
        self.idxs = idxs
        self.dataset = dataset
        
        self.epochs = args.epochs

        if args.DP == True:
            self.sample_rate = self.args.local_bs/len(idxs)
            self.noise_multiplier = args.noise_multiplier
            # self.max_grad_norm = args.clipthr*np.sqrt(args.p)
            self.max_grad_norm = args.clipthr
            self.delta = args.delta
            self.alpha_list = list(np.arange(1.01, 20.0, 0.02))    
            self.target_epsilon = args.privacy_budget
        
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.NLLLoss()
        if self.act == 'train' or self.act == 'val':
            self.ldr_train, self.ldr_val = self.split_train_test(dataset, list(idxs))
        else:
            self.ldr_test = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=False)

    def split_train_test(self, dataset, idxs):
        # split train, and test
        idxs_train = idxs[0:int((1-self.args.ratio_val)*len(idxs))]
        idxs_val = list(set(idxs)-set(idxs_train))
        train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        val = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=len(idxs_val), shuffle=False)

        return train, val

    def random_batch(self, dataset, idxs):
        idxs = list(idxs)
        idxs_train = random.sample(idxs,self.args.local_bs)
        train = DataLoader(DatasetSplit(dataset, idxs_train),
                    batch_size=self.args.local_bs,
                    shuffle=True)
        return train 
    
    
    def update_weights(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        

        if self.args.DP:
            

            privacy_engine = PrivacyEngine(
                net,
                sample_rate=self.sample_rate,
                alphas=self.alpha_list,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                secure_rng = False,
                target_epsilon=self.target_epsilon,
                target_delta = self.delta,
                epochs = self.args.local_ep*self.epochs,
            )
            privacy_engine.attach(optimizer) # With DP-SGD 
                       

        
        total_norm_list = []
        epoch_acc, epoch_loss = [], []
        for iter in range(self.args.local_ep):
            batch_acc, batch_loss = [], []
            
            if self.args.batch_type == 'mini-BSGD':
                for batch_idx, (images, labels) in enumerate(self.ldr_train): 
         
                    
                    if self.args.gpu != -1:
                        images, labels = images.cuda(), labels.cuda()
                        images, labels = autograd.Variable(images), autograd.Variable(labels)
                
                    #net.zero_grad()
                    optimizer.zero_grad()
                    
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)

                    loss.backward()          
                    
                    optimizer.step()
                    
                    y_pred = F.softmax(log_probs, dim=1).argmax(dim=1)
                    correct = torch.eq(y_pred, labels).sum().item()
                    batch_acc.append(correct/len(labels))     
                    
                    batch_loss.append(loss.item())
                
            elif self.args.batch_type == 'BSGD':

                self.ldr_train = self.random_batch(self.dataset, self.idxs)
                # w_ = deepcopy(net.state_dict())
                for batch_idx, (images, labels) in enumerate(self.ldr_train):

                    # time_list = []
                    # start_time = time.time()           
                    
                    if self.args.gpu != -1:
                        images, labels = images.cuda(), labels.cuda()
                        images, labels = autograd.Variable(images), autograd.Variable(labels)
                
                    #net.zero_grad()
                    optimizer.zero_grad()
                    
                    log_probs = net(images)
                    
                    loss = self.loss_func(log_probs, labels)

                    loss.backward()          

                    total_norm = 0
                    for p in net.parameters():
                        total_norm += pow(p.grad.norm(2), 2)
                    total_norm_list.append(np.sqrt(total_norm).numpy())
                                              
                    optimizer.step()
                                    
                    y_pred = F.softmax(log_probs, dim=1).argmax(dim=1)
                    correct = torch.eq(y_pred, labels).sum().item()
                    batch_acc.append(correct/len(labels))
                    
                    batch_loss.append(loss.item())                    


            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))

        avg_loss = sum(epoch_loss)/len(epoch_loss)
        avg_acc = sum(epoch_acc)/len(epoch_acc)

        w = net.state_dict()

        return w, avg_loss, avg_acc, total_norm_list
    

    def val(self, net):
        loss_list = []
        log_probs = []
        labels = []
        num_correct = 0
        num_samples = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            y_pred = np.argmax(log_probs.data, axis=1)
            num_samples += len(images)
            num_correct += sum(y_pred == labels.data).item()
            loss_list.append(loss.data.item())
        acc = num_correct/num_samples
        loss = sum(loss_list)/len(loss_list)

        return acc, loss, num_correct
        
    def test(self, net):
        loss_list = []
        log_probs = []
        labels = []
        num_correct = 0
        num_samples = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            y_pred = np.argmax(log_probs.data, axis=1)
            num_samples += len(images)
            num_correct += sum(y_pred == labels.data).item()
            loss_list.append(loss.data.item())
        acc = num_correct/num_samples
        loss = sum(loss_list)/len(loss_list)

        return acc, loss, num_correct
        

    
