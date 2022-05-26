from diffmd.solvers import odeint_adjoint
from diffmd.utils import compute_grad

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import copy
import csv

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
                layers += [nn.Linear(width, 1)]
            else:
                layers += [nn.Linear(width, width), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=0.01)
                nn.init.constant_(m.bias,val=0)


        # HACK
        self.mass = 1.0

    def forward(self, state):
        
        with torch.set_grad_enabled(True):        
        
            v = state[0]
            q = state[1]
            
            q.requires_grad = True
            u = self.net(q)
            
            f = -compute_grad(inputs=q, output=u)
            
            dvdt = f / self.mass
            dqdt = v
            
            
        return dvdt, dqdt


def GaussianXD(q, p):
    dim = q.shape[0]
    sigma = np.ones(dim)
    V = 0.5*np.sum(q**2 / (sigma**2))
    grad = q/(sigma**2)
    return V, grad

def Shell2D(q, p):
    r0 = np.sqrt(2)
    sigma = 0.5
    r = np.sqrt(np.dot(q, q))
    V = abs(r-r0)/sigma
    
    if (r-r0) == 0 or r == 0:
        grad = np.array([0, 0])
    else:
        grad = (q*(r-r0)/(sigma*r*abs(r-r0)))
        
    return V, grad

def Wofe_Quapp(q, p):
    x = q[0]
    y = q[1]
    V = x**4 + y**4 - 2*x**2 - 4*y**2 + x*y + 0.3*x + 0.1*y
    grad = np.array([4*x**3 - 4*x + y + 0.3, 4*y**3 - 8*y + x + 0.1])
    return V, grad

def Evaluate_H(f, q,p,M_inv):
    V_o,grad = f(q,p)
    H_o = V_o + 0.5*( np.matmul(p.reshape(1,-1),np.matmul(M_inv,p.reshape(-1,1)))   )
    H_o = H_o.reshape(-1)
    return H_o

def Neural_dynamics(f,q_temp,p_temp,M,M_inv,steps,eps,store=True):
    dim = q_temp.shape[0]
    
    batch_t = torch.linspace(0.0,eps*(steps-1),steps).to(device).type(torch.float32)
    
    q_temp = torch.tensor(q_temp).to(device).type(torch.float32)
    p_temp = torch.tensor(p_temp).to(device).type(torch.float32)


    init = (p_temp, q_temp)

    pred_y = odeint_adjoint(func, init, batch_t, method='NVE')
    
    q = pred_y[1].detach().cpu().numpy()
    p = pred_y[0].detach().cpu().numpy()

    energies = []
    
    for i in range(len(q)):
        q_i = q[i]
        p_i = p[i]

        
        curr_e = Evaluate_H(f,q_i,p_i,M_inv)
    
        energies.append(curr_e)
    
    energies = np.asarray(energies)
    
    return (q[-1], p[-1]), torch.cat(pred_y, dim=1), energies

def Neural_HMC(f,q0,num_samples,eps,steps,store=False):
    
    samples = []
    accept_rate = 0
    q = q0
    
    if store == True:
        stored_vals = np.zeros((num_samples,steps,twice_dim))
        energies = np.zeros((num_samples,steps))
        
    ind = 0
    M = np.identity(twice_dim//2)
    M_inv = np.linalg.inv(M)
    while len(samples) < num_samples:
        
        ####### Need to fix
        mean = np.zeros((twice_dim//2))
        cov = M
        p = np.random.multivariate_normal(mean, cov)
        
        q_temp = copy.deepcopy(q)
        p_temp = copy.deepcopy(p)
        
        (q_f,p_f),path,energy = Neural_dynamics(f,q_temp,p_temp,M,M_inv,steps,eps,store=True)
        stored_vals[ind,:,:] = np.asarray(path.cpu().detach().numpy())
        energies[ind,:] = np.asarray(energy).reshape(-1)
        
        p_f *= -1
        
        V_o,grad = f(q,p)
        V_f,grad = f(q_f,p_f)
        
        H_o = V_o + 0.5*np.matmul(p.reshape(1,-1),np.matmul(M_inv,p.reshape(-1,1)))
        H_o = H_o.reshape(-1)
        H_f = V_f + 0.5*np.matmul(p_f.reshape(1,-1),np.matmul(M_inv,p_f.reshape(-1,1)))
        H_f = H_f.reshape(-1)
        
        acceptance = np.exp(H_o - H_f)
        
        val = np.random.rand()

        if val < acceptance:
            q = q_f
            accept_rate += 1
            print('accepted')
        else:
            print('not accepted')
        
        samples.append(q)
        
        if len(samples)%500 == 0 and len(samples)< 2000:
            if len(samples) == 500:
                recent_samples = np.asarray(samples)
            else:
                recent_samples = np.asarray(samples[-500:])
            if (np.abs(np.linalg.det(np.cov(recent_samples.T))) > 0.001):
                M_inv = np.cov(recent_samples.T)
                M = np.linalg.inv(M_inv)
            else:
                M = np.identity(twice_dim//2)
                M_inv = np.linalg.inv(M)
#             print(M_inv)
        ind+=1
             
    acceptance = accept_rate/num_samples
#     print(stored_vals.shape)
#     print(energies.shape)
    return samples,stored_vals,energies, acceptance



twice_dim = 4
potential_function = Wofe_Quapp
num_samples = [100, 200, 500, 1000, 2000, 5000, 10000]
traj_length = 10
traj_step_size = 0.1
potential = 'wofe_quapp'
num_of_runs = 5 
num_of_models = 5

def modelled_potential(q,p):
    V = func.net(torch.Tensor(q).to(device)).item()
    grad = 0
    return V, grad

for model in range(1, num_of_models+1):
    func = torch.load(f'results/{potential}/{model}_model.pt').to(device)
    for q in num_samples:
        for i in range(1, num_of_runs+1):
            init = np.random.randn(twice_dim//2)*2
            # explicit
            start = time.perf_counter()
            samps,trajs,energies,acceptance = Neural_HMC(potential_function, init, q, traj_step_size, traj_length, store=True)
            
            if not os.path.exists(f'results/{potential}/neural_hmc_exp/model_{model}/{q}'):
                os.makedirs(f'results/{potential}/neural_hmc_exp/model_{model}/{q}')

            with open(f'results/{potential}/neural_hmc_exp/model_{model}/{q}/{i}_hmc_samps.npy', 'wb') as f:
                np.save(f, samps)

            with open(f'results/{potential}/neural_hmc_exp/model_{model}/{q}/{i}_info.npy', 'wb') as f:
                np.save(f, np.array([time.perf_counter() - start, acceptance]))

            # implicit
            init = np.random.randn(twice_dim//2)*2
            start = time.perf_counter()
            samps,trajs,energies,acceptance = Neural_HMC(modelled_potential, init, q, traj_step_size, traj_length, store=True)
            
            if not os.path.exists(f'results/{potential}/neural_hmc_mod/model_{model}/{q}'):
                os.makedirs(f'results/{potential}/neural_hmc_mod/model_{model}/{q}')

            with open(f'results/{potential}/neural_hmc_mod/model_{model}/{q}/{i}_hmc_samps.npy', 'wb') as f:
                np.save(f, samps)

            with open(f'results/{potential}/neural_hmc_mod/model_{model}/{q}/{i}_info.npy', 'wb') as f:
                np.save(f, np.array([time.perf_counter() - start, acceptance]))