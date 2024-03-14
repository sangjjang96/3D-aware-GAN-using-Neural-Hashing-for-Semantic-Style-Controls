import json
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import tinycudann as tcnn

# class Encode(nn.Module):
#     def __init__(self, in_channel, hidden_channel, out_channel, num_seg):
#         super().__init__()

#         sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#         with open("config_w.json") as f:
#             self.config = json.load(f)

#         self.network = torch.nn.Sequential(
#                 torch.nn.Conv1d(in_channel, hidden_channel, 1),
#                 torch.nn.Conv1d(hidden_channel, 64, 1),
#                 torch.nn.Conv1d(64, out_channel, 1)
#         )

#         # self.num_seg = num_seg
#         # self.seg_nets = torch.nn.ModuleList()
#         # for _ in range(num_seg):
#         #     self.seg_nets.append(torch.nn.Conv1d(64, out_channel, 1))

#     def forward(self, z):
#         # z_enc = torch.sigmoid(self.network(z)) + torch.normal(torch.Tensor([0]), torch.Tensor([1/((4*512)**2)]), device=z.device)
#         z_enc = torch.sigmoid(self.network(z))

#         # z_enc_l = []
        
#         # for i in range(self.num_seg):
#         #     z_enc_l.append(torch.sigmoid(self.seg_nets[i](z_enc)) + 1e-8)

#         return z_enc

class BRIGHT(nn.Module):
    def __init__(self, num_seg, latent):
        super().__init__()

        # self.Encoder = Encode(256, 128, 4, num_seg)
        self.num_seg = num_seg

        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        with open("config_w.json") as f:
            self.config = json.load(f)
        
        self.encodings = tcnn.Encoding(256, self.config["encoding"])

        self.networks = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.LeakyReLU(), torch.nn.Linear(latent//2, latent))
    
    def forward(self, z):        
        # z_enc = self.Encoder(z.unsqueeze(2))
        # w_hashed = self.encodings(z_enc.squeeze(2)).to(torch.float32)
        w_hashed = self.encodings(z).to(torch.float32)
        latent = self.networks(w_hashed)
        # latent = w_hashed
        
        return latent