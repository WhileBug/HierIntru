import numpy as np
import torch
import argparse
import logging
import os
import sys
import csv, random
from collections import defaultdict
from numpy.random import dirichlet
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
from tqdm import tqdm

class BasicDataset(Dataset):
    """Create the BasicDataset"""
    
    def __init__(self, data):
        super(BasicDataset, self).__init__()
        self.data = data
        self.data_length = len(data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, i):
        data = np.array(np.delete(self.data[i], -1))
        data = np.concatenate([data, np.array([0] * 8)])
        data = data.reshape(6, -1)
        data = np.expand_dims(data, axis=2)
        data = data.transpose((2, 0, 1))  # HWC to CHW
        label = np.array(self.data[i], dtype=np.float32)[-1]
        return {'data': torch.from_numpy(data), 'label': label}

def preprocess(data_path):
    """Read and preprocess the data"""
    csv_reader = csv.reader(open(data_path))
    datasets = []
    for row in csv_reader:
        data = []
        for char in row:
            if char == 'None':
                data.append(0)
            else:
                data.append(np.float32(char))
        datasets.append(data)
    # Delete ignored columns 9-21
    ignored_col = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    datasets = np.delete(datasets, ignored_col, axis=1)
    print('Data preprocessing completed!')
    return datasets

def split_data_to_clients(datasets, client_num, noniid_alpha):
    """Split the dataset into client-specific datasets based on Dirichlet distribution"""
    data_length = len(datasets)
    labels = np.array(datasets)[:, -1]
    idxs_by_class = defaultdict(list)
    # Group the data indexes by their labels
    for idx, label in enumerate(labels):
        idxs_by_class[label].append(idx)

    # Use Dirichlet distribution to create non-IID splits
    client_data_indexes = [[] for _ in range(client_num)]
    for label, idxs in idxs_by_class.items():
        proportions = dirichlet([noniid_alpha] * client_num)
        proportions = (len(idxs) * proportions).astype(int)
        current = 0
        for client_idx, num_samples in enumerate(proportions):
            client_data_indexes[client_idx].extend(idxs[current:current + num_samples])
            current += num_samples

    client_datasets = []
    for client_idx in range(client_num):
        client_data = [datasets[idx] for idx in client_data_indexes[client_idx]]
        client_datasets.append(BasicDataset(client_data))
    
    return client_datasets

def create_fl_datasets(data_path, client_num, noniid_alpha):
    """Create federated learning datasets for each client"""
    datasets = preprocess(data_path)
    client_datasets = split_data_to_clients(datasets, client_num, noniid_alpha)
    return client_datasets

def eval_net(net, loader, device, batch_size):
    """Evaluation model on test dataset"""
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

def train_net(net, client_datasets, device, epochs=5, batch_size=512, lr=0.0001, save_cp=True):
    for client_idx, client_dataset in enumerate(client_datasets):
        logging.info(f'Starting training for client {client_idx}')
        train_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        n_train = len(train_loader)
        writer = SummaryWriter(comment=f'Client_{client_idx}_LR_{lr}_BS_{batch_size}')
        global_step = 0

        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=20)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            net.train()

            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Client {client_idx} Epoch {epoch + 1}/{epochs}', unit=' record') as pbar:
                for batch in train_loader:
                    data = batch['data']
                    true_labels = batch['label']

                    data = data.to(device=device, dtype=torch.float32)
                    true_labels = true_labels.to(device=device, dtype=torch.float32)

                    labels_pred = net(data)

                    loss = criterion(labels_pred, true_labels.long())

                    epoch_loss += loss.item()

                    writer.add_scalar('Loss/train', loss.item(), global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(data.shape[0])
                    global_step += 1

            if save_cp:
                try:
                    os.mkdir('checkpoints')
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(), f'checkpoints/Client_{client_idx}_CP_epoch{epoch + 1}.pth')
                logging.info(f'Client {client_idx} Checkpoint {epoch + 1} saved!')

        writer.close()

# Example usage
data_path = 'dataset/train_data.csv'
client_num = 5
noniid_alpha = 0.5
fl_datasets = create_fl_datasets(data_path, client_num, noniid_alpha)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    from net import Net
    net = Net(input_dim=28, hidden_1=1024, hidden_2=512, hidden_3=256, out_dim=5)
    net.to(device=device)

    train_net(net=net, client_datasets=fl_datasets, epochs=5, batch_size=512, lr=0.0001, device=device)/