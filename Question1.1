#实现RNN网络架构

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_xh = nn.Linear(input_size, hidden_size)  
        self.W_hh = nn.Linear(hidden_size, hidden_size) 
        self.W_hy = nn.Linear(hidden_size, output_size)  

    def forward(self, x, h_prev):
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        y = self.W_hy(h)
        return y, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
