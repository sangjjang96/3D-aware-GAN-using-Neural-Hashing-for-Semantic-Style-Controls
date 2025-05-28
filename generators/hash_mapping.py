import json
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import tinycudann as tcnn

class Hash_Mapping(nn.Module):
    def __init__(self, num_seg, latent):
        super().__init__()

        # self.Encoder = Encode(256, 128, 4, num_seg)
        self.num_seg = num_seg

        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        with open("config_w.json") as f:
            self.config = json.load(f)
        
        # self.encoding = tcnn.Encoding(256, self.config["encoding"], dtype=torch.float32)

        self.encodings = nn.ModuleList()
        self.networks = nn.ModuleList()

        for i in range(4):
            self.encodings.append(tcnn.Encoding(4, self.config["encoding"], dtype=torch.float32))
        
        self.network = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.LeakyReLU(), torch.nn.Linear(256, latent))

    
    def forward(self, z):       
        z = torch.sigmoid(z)
        # w_hashed = self.encoding(z)
        w_hashed = self.encodings[0](z[:, :4])

        for i in range(1, 4):
            w_hashed_ = self.encodings[i](z[:, i*4:(i+1)*4])
            w_hashed = torch.cat([w_hashed, w_hashed_], -1)

        latent = self.network(w_hashed)
        # latent = w_hashed
        
        return latent