#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments   
    parser.add_argument('--epochs', type=int, default=30, help="rounds of training")
    parser.add_argument('--set_epochs', type=list, default=[45,50], help="list of training epochs")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: N")
    parser.add_argument('--frac', type=float, default=0.25, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--num_items_train', type=int, default=3000, help="dataset size for each user")
    parser.add_argument('--num_items_test', type=int, default=10000, help="dataset size for each user") 
    parser.add_argument('--ratio_val', type=float, default=0.2, help="ratio of validation dataset")      
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--lr_orig', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_g_orig', type=float, default=0.05, help='global learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='momentum parameter')
    parser.add_argument('--beta_2', type=float, default=0.99, help='momentum parameter')    
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.9)')
    parser.add_argument('--iid', type=bool, default=True, help='whether i.i.d. or not')
    parser.add_argument('--ratio_train', type=list, default=[1], help="distribution of training datasets")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='FashionMNIST', help="name of dataset")
    parser.add_argument('--data_dir', type=str, default='./datasets', help="dir of datasets")
    parser.add_argument('--batch_type', type=str, default='BSGD', help="BSGD or mini-BSGD")
    parser.add_argument('--set_algo_type', type=list, default=['Fed-SPA','Fed-RDP'], help="The selected algorithm")
    parser.add_argument('--acceleration', type=bool, default=False)

    # differential privacy
    # Two mudules: 1) with a given privacy budget and training epochs, we calculate the noise multiplier
    #              2) with a given noise multiplier, we calculate the privacy budget along with the training
    parser.add_argument('--DP', type=bool, default=True, help='whether differential privacy or not')
    parser.add_argument('--eps_cumulative', type=bool, default=False)
    parser.add_argument('--privacy_budget', type=int, default=3.0, help='The value of epsilon')
    parser.add_argument('--noise_multiplier', type=float, default=2.0, help='The noise multiplier') 
    parser.add_argument('--set_noise_multiplier', type=list, default=[0.8], help='The list of the noise multiplier')     
    parser.add_argument('--clipthr', type=int, default=5, help='The clipping threshold')
    parser.add_argument('--set_clipthr', type=list, default=[1.0], help='The set of clipping thresholds')
    parser.add_argument('--delta', type=float, default=1e-3, help='The parameter of DP')
 
    # other arguments
    parser.add_argument('--lr_decay', default=True, help="Learning rate decay")
    parser.add_argument('--lr_decay_rate', default=0.98)
    parser.add_argument('--num_experiments', type=int, default=1, help="number of experiments")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=-1)

    
    args = parser.parse_args()
    
    # For p=0.05, noise_multiplier=0.4
    # For p=1, noise_multiplier=0.4
    
    if args.dataset == 'mnist':
        args.set_num_users = [20]
        args.local_ep = 60
        args.local_bs = 10
        args.lr_orig = 0.01
        args.lr_g_orig = 0.01
        args.set_noise_multiplier = [1.4]
    elif args.dataset == 'FashionMNIST':
        args.lr_orig = 0.05
        args.local_bs = 10  
        args.lr_decay_rate = 0.95
        args.delta = 1e-3
        args.set_noise_multiplier = [1.4]
        args.set_num_users = [20]
        args.local_ep = 60
    elif args.dataset == 'cifar':
        args.lr_decay_rate = 0.98
        args.set_num_users = [20]
        args.set_noise_multiplier = [1.4]
    elif args.dataset == 'femnist':
        args.lr_decay_rate = 0.99
        args.set_num_users = [20]
        args.local_ep = 300
        args.local_bs = 4
        args.lr_orig = 0.01
        args.lr_g_orig = 0.01   
        args.set_noise_multiplier = [1.4]
    return args