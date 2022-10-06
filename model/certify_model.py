import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.ibp import *

class Certify_Model(nn.Module):
    def __init__(self):
        super(Certify_Model, self).__init__()
    
    def query():
        return    

class Certify_CNN_1D(Certify_Model):
    def __init__(self, embedding_dim, hidden_size = 100, kernel_size = 5, pool = 'mean', dropout = 0.1):
        super(Certify_CNN_1D, self).__init__()
        cnn_padding = (kernel_size - 1) // 2  # preserves size
        self.embedding_dim = embedding_dim
        self.linear_input = Linear(embedding_dim, hidden_size)
        self.conv1 = Conv1d(hidden_size, hidden_size, kernel_size,
                              padding=cnn_padding)
        self.dropout = Dropout(dropout)
        self.fc_hidden = Linear(hidden_size, hidden_size)
        self.pool = pool
        
    def forward(self, batch):
        ibp_input = batch['ibp_input']
        mask = batch['mask'] # [B, n]
        lengths = torch.sum(mask, dim=-1)
        x_h = self.linear_input(ibp_input) #[B, n, h]
        x_h = activation(F.relu, x_h) 
        x_cnn_in = x_h.permute(0, 2, 1) # [B, h, n]
        x_cnn_in = activation(F.relu, x_cnn_in)
        if self.pool == 'mean':
            fc_in = sum(x_cnn_in/ lengths.to(dtype=torch.float).view(-1, 1, 1), 2)  # B, h
            
        fc_in = self.dropout(fc_in)
        fc_hidden = activation(F.relu, self.fc_hidden(fc_in)) # B, h
        return fc_hidden
        
    