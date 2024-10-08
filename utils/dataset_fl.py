import numpy as np
import torch
from torch.utils.data import Dataset
import csv

class FederatedDataset(Dataset):
    def __init__(self, data_path='dataset/train_data.csv', train_flag=True, num_clients=5, alpha=0.5):
        super(FederatedDataset, self).__init__()
        self.train_flag = train_flag
        self.data_path = data_path
        self.data, self.data_length = self.preprocess(data_path=self.data_path, train_flag=self.train_flag)
        self.num_clients = num_clients
        self.alpha = alpha
        self.client_data = self.simulate_non_iid()

    def simulate_non_iid(self):
        """Simulate non-IID data distribution among clients"""
        num_classes = 5  # 假设有 5 个类别
        client_data_indices = []
        num_clients = self.num_clients
    
        for client in range(num_clients):
            client_data_indices.append([])
            # 从 Dirichlet 分布中生成一个样本，并进行比例调整
            proportions = np.random.dirichlet(np.ones(num_classes))
            total_samples = len(self.data) // num_clients  # 每个客户端的数据总数
    
            for class_idx in range(num_classes):
                class_indices = np.where(self.data[:, -1] == class_idx)[0]
                client_data_size = int(proportions[class_idx] * total_samples)  # 根据比例计算样本大小
    
                # 确保不会请求超过可用样本的大小
                if len(class_indices) < client_data_size:
                    client_data_size = len(class_indices)
    
                client_data_indices[client].extend(np.random.choice(class_indices, client_data_size, replace=False))
    
        return client_data_indices



    def __len__(self):
        return len(self.client_data)

    def __getitem__(self, client_idx):
        indices = self.client_data[client_idx]
        return self.data[indices]

    def preprocess(cls, data_path, train_flag):
        csv_reader = csv.reader(open(data_path))
        datasets = []
        if train_flag:
            label0_data = []
            label1_data = []
            label2_data = []
            label3_data = []
            label4_data = []
            label_status = {}
            for row in csv_reader:
                data = []
                for char in row:
                    if char=='None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))       # transform data from format of string to float32

                if data[-1] == 0:
                    label0_data.append(data)
                if data[-1] == 1:
                    label1_data.append(data)
                if data[-1] == 2:
                    label2_data.append(data)
                if data[-1] == 3:
                    label3_data.append(data)
                if data[-1] == 4:
                    label4_data.append(data)
                # record the number of different labels
                if label_status.get(str(int(data[-1])),0)>0:
                    label_status[str(int(data[-1]))] += 1
                else:
                    label_status[str(int(data[-1]))] = 1
            while len(label0_data) < 10000:
                label0_data = label0_data + label0_data
            #label0_data = random.sample(label0_data, 10000)
            while len(label1_data) < 10000:
                label1_data = label1_data + label1_data
            #label1_data = random.sample(label1_data, 10000)
            while len(label2_data) < 10000:
                label2_data = label2_data + label2_data
           # label2_data = random.sample(label2_data, 10000)
            while len(label3_data) < 10000:
                label3_data = label3_data + label3_data
           # label3_data = random.sample(label3_data, 10000)
            while len(label4_data) < 10000:
                label4_data = label4_data + label4_data
           # label4_data = random.sample(label4_data, 10000)'''
            datasets = label0_data+label1_data+label2_data+label3_data+label4_data
        else:
            for row in csv_reader:
                data = []
                for char in row:
                    if char=='None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))       # transform data from format of string to float32
                datasets.append(data)
        #delete ignored 9-21
        ignored_col = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19,20,21]
        datasets = np.delete(datasets, ignored_col, axis=1)
        data_length = len(datasets)

        minibatch = datasets
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch,dtype=np.int32)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001     # avoid getting devided by 0
        res = (data - mmin) / (mmax - mmin)
        #print('res.shape:', res.shape)
        res = np.c_[res,labels]
        np.random.shuffle(datasets)
        print('init data completed!')
        return datasets, data_length