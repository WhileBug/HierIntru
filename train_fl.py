# -*- coding:utf-8 -*-
#@Time  : 2020/6/6 15:16
#@Author: DongBinzhi
#@File  : train.py

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from net import Net
from utils.dataset_fl import FederatedDataset  # 修改为导入 FederatedDataset

dir_train_data = 'dataset/train_data.csv'
dir_test_data = 'dataset/test_data.csv'
dir_checkpoint = 'checkpoints/'

def eval_net(net, loader, device, batch_size):
    """Evaluate model on test dataset"""
    net.eval()
    n_val = len(loader)  # the number of batch
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    val_acc = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            data, true_labels = batch['data'], batch['label']
            data = data.to(device=device, dtype=torch.float32)
            true_labels = true_labels.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                labels_pred = net(data)

            loss = criterion(labels_pred, true_labels.long())
            val_loss += loss.data.item() * true_labels.size(0)
            _, pred = torch.max(labels_pred, 1)
            num_correct = (true_labels == pred).sum()
            val_acc += num_correct.item()
            pbar.update()

    net.train()
    return val_loss / (n_val * batch_size), val_acc / (n_val * batch_size)

def federated_train(net, client_datasets, device, epochs=5, batch_size=512, lr=0.0001):
    """Train the model in a federated manner"""
    for client_idx, dataset in enumerate(client_datasets):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=len(train_loader), desc=f'Client {client_idx} - Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
                for batch in train_loader:
                    data = batch['data'].to(device=device, dtype=torch.float32)
                    true_labels = batch['label'].to(device=device, dtype=torch.float32)

                    optimizer.zero_grad()
                    labels_pred = net(data)
                    loss = criterion(labels_pred, true_labels.long())
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

            val_loss, val_acc = eval_net(net, train_loader, device, batch_size)
            logging.info(f'Client {client_idx} - Epoch {epoch + 1} - Loss: {epoch_loss / len(train_loader)}, Accuracy: {val_acc}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the DNN on KDD Cup 1999.  Note: the default parameters are not the best!!!',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=512,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-n', '--num-clients', metavar='N', type=int, default=5,
                        help='Number of clients for federated learning', dest='num_clients')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Net(input_dim=28, hidden_1=1024, hidden_2=512, hidden_3=256, out_dim=5)
    print(net)
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    # 创建联邦数据集
    client_datasets = [FederatedDataset(data_path=dir_train_data, train_flag=True, num_clients=args.num_clients) for _ in range(args.num_clients)]
    
    try:
        federated_train(net=net,
                        client_datasets=client_datasets,
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batchsize,
                        lr=args.lr)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)