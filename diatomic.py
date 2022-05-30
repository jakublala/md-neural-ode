from diffmd.solvers import odeint_adjoint
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f'Using {device} device')

class ODEFunc(nn.Module):
    def __init__(self, dim, width, depth):
        super(ODEFunc, self).__init__()
        self.dim = dim
        
        layers = []
        for i in range(depth):
            if i == 0:
                layers += [nn.Linear(self.dim, width), nn.Sigmoid()]
            if i == (depth-1):
                # TODO: is the last layer of a width 1?
                layers += [nn.Linear(width, self.dim)]
            else:
                layers += [nn.Linear(width, width), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,val=0)


        # HACK
        m = 1.0
        self.mass = m*m/(m+m)

    def forward(self, state):
        
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            
            q.requires_grad = True
            
            f = -self.compute_grad(q, 0.0001)
            
            dvdt = f / self.mass
            dqdt = v
            
        return (dvdt, dqdt)

    def compute_grad(self, q, dq):
        # using this over compute_grad from nff seems to make no difference
        # HACK: does this only work with q_n => n=1? 
        # HACK: Is this wrong? Why is my NN output two values?
        return (self.net(q+dq) - self.net(q-dq)) / (2 * dq)
        
num_models = 5
dt = 0.1 # how is dt defined?

batch_size = 1000

data_model = []
data_explicit = []

for m in range(1, num_models+1):
    func = torch.load(f'results/spring/{m}_model.pt').to(device)
    
    state = torch.rand((batch_size, 2)).to(device)
    
    state = torch.split(state, 1, dim=1)

    with torch.no_grad():
        sampleLength = 10000
        batch_t = torch.linspace(0.0,dt*(sampleLength-1),sampleLength).to(device)
        state = odeint_adjoint(func, state, batch_t, method='NVE')
        state = torch.cat(state, dim=-1)
        
np.save('results/spring/diatomic_data.npy', state.cpu().numpy())