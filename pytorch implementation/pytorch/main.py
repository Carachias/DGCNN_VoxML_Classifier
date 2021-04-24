#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, FT10, FT11
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from singfpass import classify_folder
import matplotlib
import matplotlib.pyplot as plt


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    ft_loader = DataLoader(FT10(num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    for each in ft_loader:
      print(each)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    #singfpass(args, io) pass to test while training the model

    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            #print(data.shape, label.shape, logits.shape, preds.shape)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            #print(data.shape, label.shape, logits.shape, preds.shape)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        
        #########################
        # Add Selfmade Test Here
        #########################

        '''
        ft_loss = 0.0
        count = 0
        model.eval()
        ft_pred = []
        ft_true = []
        for data , label in ft_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            ft_loss += loss.item() * batch_size
            ft_true.append(label.cpu().numpy())
            ft_pred.append(preds.detach().cpu().numpy())
            print(data.shape, label.shape, logits.shape, preds.shape)
            print('LABELS:', label)
            print('PREDS:', preds)
            print('LOGITS:', logits)
        ft_true = np.concatenate(ft_true)
        ft_pred = np.concatenate(ft_pred)
        ft_acc = metrics.accuracy_score(ft_true, ft_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(ft_true, ft_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              ft_loss*1.0/count,
                                                                              ft_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        load_a_file(model)
        '''


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

def loadpretrainedtoclassify(args, io):
    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("pretrained/model.1024.t7"))
    model = model.eval()
    classify_folder(model)

def loadcustomtoclassify(args, io):
    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("pretrained/custommodel.t7"))
    model = model.eval()
    classify_folder(model)

def startmixedtesting(args, io):
    ft_loader = DataLoader(FT11(num_points=args.num_points), num_workers=8,
                           batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("pretrained/model.1024.t7"))
    model = model.eval()
    
    criterion = cal_loss
    epoch = 1


    ft_loss = 0.0
    count = 0
    model.eval()
    ft_pred = []
    ft_true = []
    for data , label in ft_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        ft_loss += loss.item() * batch_size
        ft_true.append(label.cpu().numpy())
        ft_pred.append(preds.detach().cpu().numpy())
        print(data.shape, label.shape, logits.shape, preds.shape)
        print('LABELS:', label)
        print('PREDS:', preds)
        #print('LOGITS:', logits)
    ft_true = np.concatenate(ft_true)
    ft_pred = np.concatenate(ft_pred)
    ft_acc = metrics.accuracy_score(ft_true, ft_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(ft_true, ft_pred)
    outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  ft_loss*1.0/count,
                                                                                  ft_acc,
                                                                                  avg_per_class_acc)
    io.cprint(outstr)

def startcustomtraining(args, io):
    ft_loader = DataLoader(FT10(num_points=args.num_points), num_workers=8,
                           batch_size=args.test_batch_size, shuffle=True, drop_last=True)
    ft_test_loader = DataLoader(FT11(num_points=args.num_points), num_workers=8,
                           batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    best_ft_test_acc = 0.0

    i = 0
    train_accs = []
    test_accs = []
    epochs = []


    for epoch in range(args.epochs):
        i += 1
        scheduler.step()
        ft_loss = 0.0
        count = 0
        model.train()
        ft_pred = []
        ft_true = []
        for data , label in ft_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            ft_loss += loss.item() * batch_size
            ft_true.append(label.cpu().numpy())
            ft_pred.append(preds.detach().cpu().numpy())
            #print(data.shape, label.shape, logits.shape, preds.shape)
            #print('LABELS:', label)
            #print('PREDS:', preds)
            #print('LOGITS:', logits)
        ft_true = np.concatenate(ft_true)
        ft_pred = np.concatenate(ft_pred)
        ft_acc = metrics.accuracy_score(ft_true, ft_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(ft_true, ft_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                  ft_loss*1.0/count,
                                                                                  ft_acc,
                                                                                  avg_per_class_acc)
        io.cprint(outstr)
        train_accs.append(ft_acc)
        
        
        ft_test_loss = 0.0
        count = 0
        model.eval()
        ft_test_pred = []
        ft_test_true = []
        for data , label in ft_test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            ft_test_loss += loss.item() * batch_size
            ft_test_true.append(label.cpu().numpy())
            ft_test_pred.append(preds.detach().cpu().numpy())
            #print(data.shape, label.shape, logits.shape, preds.shape)
            #print('LABELS:', label)
            #print('PREDS:', preds)
            #print('LOGITS:', logits)
        ft_test_true = np.concatenate(ft_test_true)
        ft_test_pred = np.concatenate(ft_test_pred)
        ft_test_acc = metrics.accuracy_score(ft_test_true, ft_test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(ft_test_true, ft_test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                      ft_test_loss*1.0/count,
                                                                                      ft_test_acc,
                                                                                      avg_per_class_acc)
        io.cprint(outstr)
        if ft_test_acc > best_ft_test_acc:
            print('save now')
            best_ft_test_acc = ft_test_acc
            torch.save(model.state_dict(), 'pretrained/custommodel.t7')
        #torch.save(model.state_dict(), 'pretrained/custommodel.t7')
        
        epochs.append(i)
        test_accs.append(ft_test_acc)

        fig, ax = plt.subplots()
        ax.plot(epochs, train_accs, color='blue', label = 'train acc')
        ax.plot(epochs, test_accs, color='red', label = 'test acc')
        ax.set(xlabel='epoch', ylabel='accuracy', 
              title='accuracy values per epoch')
        ax.grid()
        ax.legend()
        fig.savefig("accuracy.png")
        plt.show()


def startcustomtesting(args, io):
    ft_loader = DataLoader(FT11(num_points=args.num_points), num_workers=8,
                           batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load("pretrained/custommodel.t7"))
    model = model.eval()
    
    criterion = cal_loss
    epoch = 1


    ft_loss = 0.0
    count = 0
    model.eval()
    ft_pred = []
    ft_true = []
    for data , label in ft_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        ft_loss += loss.item() * batch_size
        ft_true.append(label.cpu().numpy())
        ft_pred.append(preds.detach().cpu().numpy())
        #print(data.shape, label.shape, logits.shape, preds.shape)
        print('LABELS:', label)
        print('PREDS:', preds)
        #print('LOGITS:', logits)
    ft_true = np.concatenate(ft_true)
    ft_pred = np.concatenate(ft_pred)
    ft_acc = metrics.accuracy_score(ft_true, ft_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(ft_true, ft_pred)
    outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                  ft_loss*1.0/count,
                                                                                  ft_acc,
                                                                                  avg_per_class_acc)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--classifymixed', action='store_true', dest='classify_mode_mixed',
                        default=False, help='Classify files')
    parser.add_argument('--classifycustom', action='store_true', dest='classify_mode_custom',
                        default=False, help='Classify files')
    parser.add_argument('--customtrain', action='store_true', dest='train_custom',
                        default=False, help='Train on other data')
    parser.add_argument('--evalmixedperf', action='store_true', dest='eval_mixed',
                        default=False, help='Train on other data')
    parser.add_argument('--evalcustomperf', action='store_true', dest='eval_custom',
                        default=False, help='Train on other data')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    print(args)

    if args.eval_mixed == False:
        pass
    else:
        #test pretrained model with self converted files
        startmixedtesting(args, io)


    if args.train_custom == False:
        pass
    else:
        #start training routine w custom dset
        startcustomtraining(args, io)
      

    if args.eval_custom == False:
        pass
    else:
        #test custom trained model
        startcustomtesting(args, io)


    if args.classify_mode_custom == False:
        pass
    else:
        #classify objects on custom trained model
        loadcustomtoclassify(args, io)


    if args.classify_mode_mixed == False:
        pass
    else:
        #classify objects on pretrained model
        loadpretrainedtoclassify(args, io)

    ''' unnecessary old functions for train/test on non custom ModelNet 40    
    if not args.eval:
        train(args, io)
    else:
        test(args, io)
    '''