
import os
from glob import glob
import copy
import collections
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
import torch.nn as nn
from torch import optim

import models
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weighted_averaging(w_list, num_sample_list):
    num_total_samples = sum(num_sample_list)
    keys = w_list[0].keys()
    w_avg = collections.OrderedDict()
    for k in keys:
        w_avg[k] = torch.zeros(w_list[0][k].size()).to(device)   # Reshape w_avg to match local weights.

    for k in keys:
        for i in range(len(w_list)):
            w_avg[k] += num_sample_list[i] * w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], num_total_samples)
    return w_avg

    

class SingleClassClientData(Dataset):
    def __init__(self, img_dir, label, transform=None):
        self.img_list = glob(os.path.join(img_dir, '*.JPG'))
        self.img_labels = [label] * len(self.img_list)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        # image = read_image(img_path)
        image = default_loader(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class MultiClassClientData(Dataset):
    def __init__(self, parent_dir, label_idx_dict, transform=None):
        
        self.img_list, self.label_list = [], []
        sub_dirs = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
        for sub_dir in sub_dirs:
            full_path = os.path.join(parent_dir, sub_dir)
            img_paths = glob(os.path.join(full_path, '*.JPG')) + glob(os.path.join(full_path, '*.jpg'))   
            labels = [label_idx_dict[sub_dir]] * len(img_paths)
            self.img_list += img_paths
            self.label_list += labels
            
        self.transform = transform
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = default_loader(img_path)
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def local_update(model_struct, glob_w, data_loader, num_epochs, criterion=nn.CrossEntropyLoss()):
    
    # local_model = copy.deepcopy(glob_model)
    local_model = copy.deepcopy(model_struct)
    local_model.load_state_dict(copy.deepcopy(glob_w))
    
    learning_rate = 0.001
    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

    # optimizer = optim.SGD(local_model.parameters(), lr=0.01, weight_decay=0.00001, momentum=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model.
    local_model.train()
    for epoch in range(num_epochs):
        for step, data in enumerate(data_loader):
            optimizer.zero_grad() 
            x_batch, y_batch = data[0].to(device), data[1].to(device)
            _, y_pred = local_model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Calculate local training performance.
    valid_loss, valid_acc = evaluate_model(local_model, data_loader)
    
    # Return update.
    local_update_dict ={
        'local_w': local_model.state_dict(),
        'num_samples': len(data_loader.dataset),
        'loss': valid_loss,
        'acc': valid_acc
    }
    
    return local_update_dict



def local_update_moon(model_struct, glob_w, prev_w, data_loader, num_epochs, criterion=nn.CrossEntropyLoss(), use_loss_con=True):

    glob_model = copy.deepcopy(model_struct)
    glob_model.load_state_dict(copy.deepcopy(glob_w))
    glob_model.eval()
    for param in glob_model.parameters():
        param.requires_grad = False

    prev_model = copy.deepcopy(model_struct)
    prev_model.load_state_dict(copy.deepcopy(prev_w))
    prev_model.eval()
    for param in prev_model.parameters():
        param.requires_grad = False

    local_model = copy.deepcopy(model_struct)
    local_model.load_state_dict(copy.deepcopy(glob_w))
    local_model.train()

    learning_rate = 0.001
    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(local_model.parameters(), lr=0.01, weight_decay=0.00001, momentum=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    temperature = 0.5
    mu = 3

    for epoch in range(num_epochs):
        for step, data in enumerate(data_loader):
            optimizer.zero_grad() 
            x_batch, y_batch = data[0].to(device), data[1].to(device)

            z, y_pred = local_model(x_batch)
            z_glob, _ = glob_model(x_batch)
            z_prev, _ = prev_model(x_batch)
            
            # Contrastive loss calculation. This gives the same result as below code.
            # positive = cosine_similarity(z, z_glob) 
            # logits = positive.reshape(-1,1)
            
            # negative = cosine_similarity(z, z_prev)
            # logits = torch.cat((logits, negative.reshape(-1,1)), dim=1)
            # logits /= temperature
            
            # labels = torch.zeros(x_batch.size(0)).cuda().long()
            # loss_con = criterion(logits, labels)
            
            # Contrastive loss calculation.
            postive = torch.mean(cosine_similarity(z, z_glob) / temperature)
            negative = torch.mean(cosine_similarity(z, z_prev) / temperature)
            loss_con = - torch.log(torch.exp(postive) / (torch.exp(postive) + torch.exp(negative)))
            
            loss_sup = criterion(y_pred, y_batch)
            
            if use_loss_con:
                loss = loss_sup + mu * loss_con
            else:
                loss = loss_sup
            
            loss.backward()
            optimizer.step()
    
    # Calculate local training performance.
    valid_loss, valid_acc = evaluate_model(local_model, data_loader)
    
    # Return update.
    local_update_dict ={
        'local_w': local_model.state_dict(),
        'num_samples': len(data_loader.dataset),
        'loss': valid_loss,
        'acc': valid_acc
    }
    
    return local_update_dict



def evaluate_model(model, data_loader, criterion=nn.CrossEntropyLoss()):

    loss_metric = utils.MeanMetric()
    acc_metric = utils.MeanMetric()

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_batch, y_batch = data[0].to(device), data[1].to(device)
            _, y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss_metric.update_state(loss.item())
            acc_metric.update_state(utils.compute_accuracy(y_batch, y_pred))

    return loss_metric.result(), acc_metric.result()

    


# def get_data_loader(train_dir, valid_dir=None, transform=None, batch_size=32, num_workers=2):

#     train_data = ImageFolder(train_dir, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     if valid_data:
#         valid_data = ImageFolder(valid_dir, transform=transform)
#         valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#         return train_loader, valid_loader

#     return train_loader


# def weighted_average_weights(w, num_samples):
#     w_avg = copy.deepcopy(w[0]) # The first user
#     total_size = np.sum(num_samples)

#     for k in w_avg.keys():
#         # print("w0:", w_avg[k])
#         w_avg[k] *= num_samples[0] #Weighted for the first user
#         # print("w0*Ds:", w_avg[k])
#         # print("Numb of samples:", num_samples[0])
#         # print("update weight key=", str(k))
#         for i in range(1, len(w)): # The remanining users
#             # print("Numb of samples:", num_samples[i])
#             w_avg[k] += num_samples[i]*w[i][k]
#         # print("w*Ds:", w_avg[k])
#         w_avg[k] = torch.div(w_avg[k], total_size)
#         # print("w_avg:", w_avg[k])
#     return w_avg 