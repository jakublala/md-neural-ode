from diffmd.solvers import odeint_adjoint
from diffmd.utils import compute_grad

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f'Using {device} device')

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.losses = []
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.log(val)
    
    def log(self, val):
        self.losses.append(val)

def get_first_batch(training_trajs,nsample,sampleLength,dt):
    twice_dim = training_trajs.size()[2]
    dim = twice_dim//2

    q0 = training_trajs[0:nsample, 0, :dim].to(device)
    p0 = training_trajs[0:nsample, 0, dim:].to(device)
    batch_y0 = (p0, q0)
    
    q = training_trajs[0:nsample, 0:sampleLength, :dim].to(device)
    p = training_trajs[0:nsample, 0:sampleLength, dim:].to(device)
    batch_y = torch.cat((p, q), dim=2).swapaxes(0, 1)
    
    batch_t = torch.linspace(0.0,dt*(sampleLength-1),sampleLength).to(device)
    return batch_t, batch_y0, batch_y


def get_batch_mod(traj,batch_size,batch_length,dt):
    
    twice_dim = traj.size()[2]
    dim = twice_dim//2
    sampled_is = torch.randint(traj.shape[0],size = (batch_size,)).to(device)
    sampled_js = torch.randint(traj.shape[1]-batch_length,size = (batch_size,)).to(device)
    initial_time = sampled_js*dt
    
    batch_t = torch.linspace(0.0,dt*(batch_length-1),batch_length).to(device)
    qs = traj[sampled_is,sampled_js,:dim]
    ps = traj[sampled_is,sampled_js,dim:]
    pos_init = (ps, qs)
    # print('p', ps)
    # print('q', qs)
    
    sampled_trajs = []
    
    for i in range(batch_size):
        qs = traj[sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,:dim].view(-1,dim)
        ps = traj[sampled_is[i],sampled_js[i]:sampled_js[i]+batch_length,dim:].view(-1,dim)
        x = torch.cat((ps, qs), dim=1)
        sampled_trajs.append(x)
    
        
    batch_trajs = torch.stack(sampled_trajs,dim=1)
    return batch_t,pos_init,batch_trajs

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
            
            f = -compute_grad(inputs=q, output=u.T)
            
            dvdt = f / self.mass
            dqdt = v
            
            
        return (dvdt, dqdt)

    def compute_grad(self, q, dq):
        # using this over compute_grad from nff seems to make no difference
        # HACK: does this only work with q_n => n=1? 
        return (self.net(q+dq) - self.net(q-dq)) / (2 * dq)
        

def get_data(potential, train_split):
    trajs = np.load(f'dataset/{potential}.npy')
    if train_split == 1.0:
        return torch.Tensor(trajs).to(device), None
    else:
        test_split = 1 - train_split
        split_index = int(trajs.shape[0] * train_split)
        
        np.random.shuffle(trajs)
        
        training_trajs = torch.Tensor(trajs[:split_index, :, :]).to(device)
        testing_trajs = torch.Tensor(trajs[split_index:, :, :]).to(device)
        return training_trajs, testing_trajs 

def train_model(niters, training_trajs, dt, sample_length, batch_size, learning_rate, scheduling_factor, scheduling_freq, nn_depth, nn_width):    
    loss_meter = RunningAverageMeter()
    dim = training_trajs.size()[2] // 2
    func = ODEFunc(dim, nn_width, nn_depth).to(device)
    # func = torch.load('results/2d_shell/model.pt')
    optimizer = torch.optim.Adam(func.parameters(), lr=learning_rate)
    
    lambda1 = lambda epoch: scheduling_factor ** epoch
    scheduler =  torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    if os.path.exists('temp'):
        shutil.rmtree('temp')

    if not os.path.exists('temp'):
        os.makedirs('temp')

    for itr in range(1, niters + 1):
        start = time.perf_counter()
        optimizer.zero_grad()
        
        batch_t, batch_y0, batch_y = get_batch_mod(training_trajs, batch_size, sample_length,dt)
        
        # convert momentum to velocity (v = p / mu)
        batch_y0 = (batch_y0[0] / func.mass, batch_y0[1])
        pred_y = odeint_adjoint(func, batch_y0, batch_t, method='NVE')

        # convert velocity to momentum (p = mu * v)
        pred_y = (pred_y[0] * func.mass, pred_y[1])
        pred_y = torch.cat(pred_y, dim=2)
        
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward() 
        optimizer.step()
        loss_meter.update(loss.item())
        
        if itr % 500 == 0: # output log throughout
            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, loss_meter.avg))
            print('current loss: {:.4f}'.format(loss_meter.val))
            print('Last iteration took: ', time.perf_counter() - start)

        if itr % 1000 == 0: # do a validation step across entire trajectory
            for i in range(2):
                plt.plot(pred_y[:,i,1].detach().cpu().numpy(),pred_y[:,i,0].detach().cpu().numpy(),color='blue',alpha=0.5)
                plt.plot(batch_y[:,i,1].detach().cpu().numpy(),batch_y[:,i,0].detach().cpu().numpy(),color='red',alpha=0.5)
            plt.savefig('temp/{0}.png'.format(itr))
            plt.close()

        if itr % scheduling_freq == 0:
            scheduler.step()
            print('current learning rate: ', scheduler.get_last_lr())

    return func, loss_meter

def main():
    sample_length=20
    batch_size=800
    learning_rate=0.025
    scheduling_factor=0.6
    scheduling_freq=2000
    nn_depth=2
    nn_width=50
    niters = 20000
    
    train_split = 1.0
    potentials = ['wofe_quapp']
    num_models = 5
    for potential in potentials:
        if potential == '2d_shell':
            dt = 0.01
        else:
            dt = 0.1

        for i in range(1, 1+num_models):
            training_trajs, testing_trajs = get_data(potential, train_split)
            model, loss_meter = train_model(niters, training_trajs, dt, sample_length, batch_size, learning_rate, scheduling_factor, scheduling_freq, nn_depth, nn_width)

            # save model
            torch.save(model, f'results/{potential}/{i}_model.pt')
            with open(f'results/{potential}/{i}_loss.txt', 'w') as f:
                for loss in loss_meter.losses:
                    f.write(f'{loss}\n')
        

if __name__ == '__main__':
    main()