#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import time
# import matplotlib
# matplotlib.use('Agg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import logging
import datetime
import random


from torchvision import datasets, transforms
from copy import deepcopy

from sampling import mnist_iid, cifar_iid
from options import args_parser
from update import LocalUpdate
from fednets import CNNMnist, CNNCifar, CNNFemnist
from averaging import average_weights
from calculate import subtract, add, add_cons, para_root, para_divide, para_pow, multipl_cons
from utils.dataset_preprocess import FEMNIST
from noise_compute import ComputeNoiseEpsA

"SPA Reference: Federated Learning with Sparsification-Amplified Privacy and Adaptive Optimization"
" https://www.ijcai.org/proceedings/2021/0202.pdf"

"FL-RDP Reference: https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py."

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def compute_noise_mul_spa(args, num_steps, q, target_eps):
    
    alpha_list = np.arange(1.01, 100.0, 0.02)

    rdp_list = target_eps - np.log(1/args.delta)/(np.array(alpha_list)-1)

    noise_list = 7*pow(q,2)*num_steps*np.array(alpha_list)/np.array(rdp_list)

    best_noise_mul = min(noise_list[np.where(noise_list>0)])
    best_alpha = alpha_list[np.argmin(noise_list[np.where(noise_list>0)])]

    # print('Noise list:', self.noise_list)

    return np.sqrt(best_noise_mul), best_alpha

def model_build(args):
    
    net_glob = None
    if args.dataset == 'mnist':
        if args.gpu != -1:
            net_glob = CNNMnist(args=args).cuda()
        else:
            net_glob = CNNMnist(args=args)
    elif args.dataset == 'FashionMNIST':
        # if args.gpu != -1:
        #     net_glob = CNNFashionMnist(args=args).cuda()
        # else:
        #     net_glob = CNNFashionMnist(args=args)
        if args.gpu != -1:
            net_glob = CNNMnist(args=args).cuda()
        else:
            net_glob = CNNMnist(args=args)
    elif args.dataset == 'cifar':
        if args.gpu != -1:
            net_glob = CNNCifar(args=args).cuda()
        else:
            net_glob = CNNCifar(args=args)
            
    elif args.dataset == 'femnist':
        if args.gpu != -1:
            net_glob = CNNFemnist(args=args).cuda()
        else:
            net_glob = CNNFemnist(args=args)        
        
    else:
        exit('Error: unrecognized model')
    print("Nerual Net:",net_glob)
    
    
    net_glob.train()  #Train() does not change the weight values
    # copy weights
    w_glob = net_glob.state_dict()      
    w_size = 0
    w_size_all = 0
    for k in w_glob.keys():
        size = w_glob[k].size()
        if(len(size)==1):
            nelements = size[0]
        else:
            nelements = size[0] * size[1]
        w_size += nelements*32
        w_size_all += nelements
        # print("Size ", k, ": ",nelements*4)
    print('weight element numbers:', w_size_all)
    print("Weight Size:", w_size, " bits")
    print("Weight & Grad Size:", w_size*2, " bits")
    print("Each user Training size:", 784* 8/8* args.local_bs, " bytes")
    print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
    
    return net_glob, w_glob, w_size

if __name__ == '__main__':    
    # return the available GPU
    # av_GPU = torch.cuda.is_available()
    # if  av_GPU == False:
    #     exit('No available GPU')
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    #####-Choose Variable-#####
    set_variable = args.set_clipthr
    set_variable0 = args.set_epochs
    set_variable1 = args.set_algo_type

    date = datetime.date.today()
    if not os.path.exists('./logs-{}'.format(date)):
        os.mkdir('./logs-{}'.format(date))
        
    log_time = int(time.time())
    
    log_dir = './logs-{}'.format(date)
    
    logger = get_logger(log_dir + '/log-{}.log'.format(log_time))
    
    logger.info(log_dir + '/log-{}/start training!'.format(log_time))
    logger.info(log_dir + '/log-{}/{}'.format(log_time,args))

    print(args)
    
    for num_users in args.set_num_users:

        args.num_users = num_users
        
            
        # load dataset and split users
        if args.dataset == 'mnist':
            dataset_train = datasets.MNIST(args.data_dir+'/mnist/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
            dataset_test = datasets.MNIST(args.data_dir+'/mnist/', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))    
            
            dict_train = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            dict_test = range(len(dataset_test))
            
        elif args.dataset == 'FashionMNIST':
            dataset_train = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
            dataset_test = datasets.FashionMNIST(args.data_dir, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
         
            dict_train = mnist_iid(args, dataset_train, args.num_users, args.num_items_train)
            dict_test = range(len(dataset_test))
            
            
        elif args.dataset == 'cifar':
            dict_train_train, dict_sever = {},{}
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10(args.data_dir+'/cifar/', train=True, transform=transform, target_transform=None, download=True)
            dataset_test = datasets.CIFAR10(args.data_dir+'/cifar/', train=False, transform=transform, target_transform=None, download=True)

            args.num_items_train = int(len(dataset_train)/args.num_users)
            dict_train = cifar_iid(dataset_train, args.num_users, args.num_items_train)
            dict_test = range(len(dataset_test))
            
            # dict_test = cifar_iid(dataset_test, args.num_users, args.num_items_test)
        
        elif args.dataset == 'femnist':
            dataset_train = FEMNIST(data_dir=args.data_dir,train=True)
            dataset_test = FEMNIST(data_dir=args.data_dir,train=False)
            
            dict_train = dataset_train.get_client_dic()
            dict_test = range(len(dataset_test))

        # torch.save(dict_train, './data_assign/dict_train-{}.pt'.format(args.dataset))
        # torch.save(dict_test, './data_assign/dict_test-{}.pt'.format(args.dataset))
                     
        # dict_train = torch.load('./data_assign/dict_train-{}.pt'.format(args.dataset))
        # dict_test = torch.load('./data_assign/dict_test-{}.pt'.format(args.dataset))

            
        for index0 in range(len(args.set_noise_multiplier)):
            args.noise_multiplier = args.set_noise_multiplier[index0]
            
            final_train_loss = np.zeros([len(args.set_clipthr), len(args.set_epochs), len(args.set_algo_type)])
            final_train_acc = np.zeros([len(args.set_clipthr), len(args.set_epochs), len(args.set_algo_type)])
            final_test_loss = np.zeros([len(args.set_clipthr), len(args.set_epochs), len(args.set_algo_type)])
            final_test_acc = np.zeros([len(args.set_clipthr), len(args.set_epochs), len(args.set_algo_type)])
            final_eps = np.zeros([len(args.set_clipthr), len(args.set_epochs), len(args.set_algo_type)])
            
            nums_combination = len(args.set_clipthr)*len(args.set_epochs)*len(args.set_algo_type)
            for s in range(nums_combination): 
    
                index1 = int(s/(len(args.set_epochs)*len(args.set_algo_type)))
                index2 = int((s-index1*(len(args.set_epochs)*len(args.set_algo_type)))/len(args.set_algo_type))
                index3 = int((s-index1*(len(args.set_epochs)*len(args.set_algo_type)))-index2*len(args.set_algo_type))

                args.clipthr = args.set_clipthr[index1]
                args.epochs = args.set_epochs[index2]
                args.algo = args.set_algo_type[index3]
   
                loss_test, loss_train = [], []
                acc_test, acc_train = [], []   
                test_acc_list = np.array([0 for i in range(args.epochs)])
                test_loss_list = np.array([0 for i in range(args.epochs)])
                val_acc_list = np.array([0 for i in range(args.epochs)])
                                    
                for m in range(args.num_experiments):
                    
                    # initilize the model
                    net_glob, w_glob, args.w_size = model_build(args)
                    # training
                    loss_avg_list, acc_avg_list, list_loss, loss_avg = [], [], [], []
                    acc_test_list, loss_test_list= [],[]
                    acc_val_list, com_cons = [], []
                    
        
                    ### Neural network sparse ###
                    # print('Model parameters:', w_glob) 
                    
                    if args.DP:
                        
                        scale_cg = 25
                        
                        if args.batch_type == 'mini-BSGD':
                            num_steps = args.frac*args.local_ep*args.epochs*np.floor(int((1-args.ratio_val)*args.num_items_train)/args.local_bs)
                        elif args.batch_type == 'BSGD':
                            num_steps = args.frac*args.local_ep*args.epochs
                    
                        q = args.local_bs / int((1-args.ratio_val)*args.num_items_train)
                     
                        
                        if not args.eps_cumulative:
                        ### Compute noise scale ###
                                                  
                            eps_sgd = args.privacy_budget - args.epochs/scale_cg
                            if args.algo == 'Fed-SPA':
                                args.noise_multiplier, best_alpha = compute_noise_mul_spa(args, num_steps, q, eps_sgd)
                            else:
                                comp_obj = ComputeNoiseEpsA()
                                comp_obj.delta = args.delta
                                comp_obj.alpha_list = list(np.arange(1.01, 100.0, 0.02))
                                comp_obj.q = q
                                args.noise_multiplier, eps_temp, best_alpha, temp_rdp\
                                    = comp_obj.compute_noise_mul(target_eps=eps_sgd, search_list=list( np.arange(0.1, 10, 0.1) ) )
                        
                        print('\nNoise multiplier: {}'.format(args.noise_multiplier))                        

                        logger.info(log_dir + '/log-{}/\nNoise multiplier: {}'.format(log_time,args.noise_multiplier))


                    if args.acceleration == True:
                        w_u_ = subtract(w_glob, w_glob)
                        w_v_ = add_cons(w_u_, 1e-6)    

                    client_indictor = np.array([0 for i in range(args.num_users)])
                    eps_value = [0 for i in range(args.num_users)]
                    
                    time_matrix = []
                    norm_list = []
                    
                    for iter in range(args.epochs):
                        print('*' * 20,f'Experi: {m}/{args.num_experiments}, Epoch: {iter}','*' * 20) 
                                                                                                   
                        set_users = random.sample(range(args.num_users), int(args.num_users*args.frac))
                        client_indictor[set_users] += 1                                                                                       
                       
                        # learning rate decay
                        if args.lr_decay == True:
                            args.lr = args.lr_decay_rate*args.lr_orig
                            args.lr_g = args.lr_decay_rate*args.lr_orig
                                                                            
                        # record the run time
                        time_list = []
                        
                        start_time = time.time()
                                                        
                        w_locals, loss_locals, acc_locals = [], [], []

                        for idx in range(len(set_users)):
                                                                
                            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_train[set_users[idx]], act='train')
                            w, loss, acc, norm = local.update_weights(net=deepcopy(net_glob))                     
                            w_locals.append(subtract(w, w_glob))
                            
                            if args.eps_cumulative and args.DP:  
                                alpha_list = np.arange(1.01, 100.0, 0.02)   

                                if args.batch_type == 'mini-BSGD':
                                    eps_alpha = 7*args.frac*args.local_ep*np.floor(int((1-args.ratio_val)\
                                            *args.num_items_train)/args.local_bs)*client_indictor[set_users[idx]]*alpha_list*\
                                            pow(q/args.noise_multiplier,2)+np.log(1/args.delta)/(alpha_list-1)  
                                elif args.batch_type == 'BSGD':                                    
                                    eps_alpha = 7*args.frac*args.local_ep*client_indictor[set_users[idx]]*alpha_list*\
                                            pow(q/args.noise_multiplier,2)+np.log(1/args.delta)/(alpha_list-1)                                        
                                # eps_opti = min(eps_alpha) + args.eps_cg*(iter+1)/args.epochs
                                eps_opti = min(eps_alpha) + client_indictor[set_users[idx]]/scale_cg
                                eps_value[set_users[idx]] = eps_opti
                                # print('Optimal alpha:', alpha_list[np.argmin(eps_alpha)])
                                                           
                                print('\nEps:', eps_value, '\nMaximum norm:', max(norm_list)) 
                            
                            loss_locals.append(loss) 
                            acc_locals.append(acc)
                            
                            norm_list += norm
                                                  
                        
                        end_time = time.time()
                        
                        time_list.append(end_time - start_time)
                        
                        start_time = time.time()
                        
                        # calculate training loss and training accuracy
                        if len(loss_locals) > 0:
                            
                            epoch_train_loss = sum(loss_locals) / len(loss_locals)
                            epoch_train_acc = sum(acc_locals) / len(acc_locals)                                
                            
                            print("\nTrain loss: {}, train acc: {}".format(epoch_train_loss, epoch_train_acc))



                            w_locals_pre = deepcopy(w_locals)
                                                            
                            w_update = average_weights(args, w_locals)
         
                            if args.acceleration == True:
                                w_u = add(multipl_cons(w_u_, args.beta_1), multipl_cons(w_update, (1-args.beta_1)))
                                w_v = add(multipl_cons(w_v_, args.beta_2), multipl_cons(para_pow(w_u), (1-args.beta_2)))
                                w_update = multipl_cons(para_divide(w_u, add_cons(para_root(w_v), 1e-3)), args.lr_g)
                                w_u_, w_v_ = deepcopy(w_u), deepcopy(w_v)
                                # w_update = average_weights_mask(args, w_locals, index_mask_locals)
                            w_glob = add(w_glob, w_update)

                        
                        # model validation
                        net_glob.load_state_dict(w_glob)
                        net_glob.eval()
                        
                        if args.DP:
                                                        
                            acc_list, loss_list = [], []
                            score_list, pure_score_list = [], []
                            for c in range(args.num_users):
                                net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_train[c], act='val')
                                acc, loss, score = net_local.val(net=net_glob)
                                score_list.append(sum([score, np.random.laplace(loc=0.0, scale=scale_cg, size=None)]))
                                pure_score_list.append(score)
                                acc_list.append(acc)
                                loss_list.append(loss)
                            
                            epoch_val_acc, epoch_val_loss = sum(score_list)/(args.num_users*args.ratio_val*args.num_items_train), sum(loss_list)/len(loss_list)
                        
                        else:
                            
                            acc_list, loss_list = [], []
                            for c in range(args.num_users):
                                net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_train[c], act='val')
                                acc, loss, score = net_local.val(net=net_glob)
                                acc_list.append(acc)
                                loss_list.append(loss)
                            
                            epoch_val_acc, epoch_val_loss = sum(acc_list)/len(acc_list), sum(loss_list)/len(loss_list)
                                                        
                        # model test 
                        
                        net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_test, act='test')
                        epoch_test_acc, epoch_test_loss, score = net_local.test(net=net_glob)                            
                        
                        end_time = time.time()
                        
                        time_list.append(end_time - start_time)
                        
                        time_matrix.append(time_list)


                        print("Val loss: {}, Val acc: {}".\
                                format(epoch_val_loss, epoch_val_acc))                               
                        print("Test loss: {}, test acc: {}\n".\
                                format(epoch_test_loss, epoch_test_acc))

                        print('\nRun time: {}s'.format(sum(np.array(time_matrix))/(iter+1)))  
                                
                        acc_test_list.append(epoch_test_acc)
                        loss_test_list.append(epoch_test_loss)
                        acc_val_list.append(epoch_val_acc)
                        
                        if args.algo == 'Fed-SPA' and args.DP:
                            
                            if pow(args.noise_multiplier,2)/2<0.7:
                                break
                    
                    logger.info(log_dir + '/log-{}/\nCurrent test loss: {}'.format(log_time,loss_test_list))
                    logger.info(log_dir + '/log-{}/\nCurrent test acc: {}'.format(log_time,acc_test_list))

                    norm_list.sort()
                    # logger.info(log_dir + '/log-{}/\nNorm list: {}, median value: {}'.format(log_time,norm_list,norm_list[int(len(norm_list)/2)]))
                    
                    test_acc_list = deepcopy((test_acc_list*m+np.array(acc_test_list))/(m+1))
                    test_loss_list = deepcopy((test_loss_list*m+np.array(loss_test_list))/(m+1))
                    val_acc_list = deepcopy((val_acc_list*m+np.array(acc_val_list))/(m+1))                        

                    # test_acc_list,test_loss_list = deepcopy(acc_test_list), deepcopy(loss_test_list)
                    
                    print('\nTest loss:', test_loss_list)
                    print('\nTest acc:', test_acc_list)
                    # print('\nTest loss:', loss_test_list)
                    # print('\nTest acc:', acc_test_list)
                    print('\nVal acc:', acc_val_list)
                    print('\nEps:', eps_value)

                    print('\nMedian norm:', norm_list[int(len(norm_list)/2)])
                                                

                    logger.info(log_dir + '/log-{}/\nTest loss: {}'.format(log_time,test_loss_list))
                    logger.info(log_dir + '/log-{}/\nTest acc: {}'.format(log_time,test_acc_list))
                    logger.info(log_dir + '/log-{}/\nVal acc: {}'.format(log_time,acc_val_list))
                    logger.info(log_dir + '/log-{}/\nEps: {}'.format(log_time,eps_value))
                    logger.info(log_dir + '/log-{}/\nMedian norm: {}'.format(log_time, norm_list[int(len(norm_list)/2)]))                
                    
                    loss_train.append(epoch_train_loss)
                    acc_train.append(epoch_train_acc)
                    idx_opti = np.argmax(acc_val_list)
                    loss_test.append(loss_test_list[idx_opti])            
                    acc_test.append(acc_test_list[idx_opti])
                    # loss_test.append(test_loss_list[idx_opti])            
                    # acc_test.append(test_acc_list[idx_opti])
                    
                    # loss_test.append(epoch_test_loss)                
                    # acc_test.append(epoch_test_acc)
                    
                # plot loss curve
                final_train_loss[index1][index2][index3] = deepcopy(sum(loss_train) / len(loss_train))
                final_train_acc[index1][index2][index3] = deepcopy(sum(acc_train) / len(acc_train))
                final_test_loss[index1][index2][index3] = deepcopy(sum(loss_test) / len(loss_test))
                final_test_acc[index1][index2][index3] = deepcopy(sum(acc_test) / len(acc_test))                  

                print('\nFinal train loss:', final_train_loss)
                print('\nFinal train acc:', final_train_acc)
                print('\nFinal test loss:', final_test_loss)
                print('\nFinal test acc:', final_test_acc)


                logger.info(log_dir + '/log-{}/\nFinal train loss: {}'.format(log_time,final_train_loss))                
                logger.info(log_dir + '/log-{}/\nFinal train acc: {}'.format(log_time,final_train_acc))
                logger.info(log_dir + '/log-{}/\nFinal test loss: {}'.format(log_time,final_test_loss))
                logger.info(log_dir + '/log-{}/\nFinal test acc: {}'.format(log_time,final_test_acc))