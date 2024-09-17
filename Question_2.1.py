import numpy as np
import torch

class PE(torch.nn.Module):
    def __init__(self, mod_d, max_length):
        super(PE,self).__init__()

        position_encode=np.zeros((max_length,mod_d))
        position=np.arange(0,max_length).reshape(-1,1)
        ecode_fenmu = np.exp(np.arange(0, mod_d, 2) * -(np.log(10000.0) / mod_d))\
        
        position_encode[:, 0::2] = np.sin(position * ecode_fenmu) 
        position_encode[:, 1::2] = np.cos(position * ecode_fenmu)
        position_encode=torch.tensor(position_encode,dtype=torch.float32).unsqueeze(0)

        self.register_buffer('position_encode',position_encode)

    def forward(self,x):
        return x+self.position_encode[:,:x.size(1),:]
