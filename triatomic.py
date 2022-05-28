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
            
            f = -self.compute_grad(q, 0.01)
            
            dvdt = f / self.mass
            dqdt = v
            
        return (dvdt, dqdt)

    def compute_grad(self, q, dq):
        # using this over compute_grad from nff seems to make no difference
        # HACK: does this only work with q_n => n=1? 
        # HACK: Is this wrong? Why is my NN output two values?
        return (self.net(q+dq) - self.net(q-dq)) / (2 * dq)
        

class ODEFunc_NoEndFixed():
    def __init__(self, func):
        self.r0 = 1
        
        self.pair_potential = func.net 
        self.grad = func.compute_grad
        
        self.mass = 1.0

        self.forces_log = []
        self.energies_log = []

    def system_potential(self, q):
        qs = torch.Tensor([torch.abs(q - self.q3) - self.r0, torch.abs(q - self.q1) - self.r0]).to(device).view(-1, 1)
        energy = torch.sum(self.pair_potential(qs))
        return energy

    def system_force(self, q, dq):
        grad = (self.system_potential(q+dq) - self.system_potential(q-dq)) / (2 * dq)
        return -grad

    def separate_force(self, q, dq):
        qs = ([torch.abs(q[1] - q[0]) - self.r0, torch.abs(q[2] - q[1]) - self.r0])
        grad_A = (self.pair_potential(qs[0]+dq) - self.pair_potential(qs[0]-dq)) / (2 * dq)
        grad_B = (self.pair_potential(qs[1]+dq) - self.pair_potential(qs[1]-dq)) / (2 * dq)
        grad = torch.Tensor([grad_A, -grad_A + grad_B, -grad_B]).view(-1, 1).to(device)
        return grad


    def integrate_verlet(self, state, dt):
        v,q = copy.deepcopy(state)
        assert q[1] > q[0], 'the two atoms collided'
        
        force = self.separate_force(q, 0.001)
        self.forces_log.append(force.cpu().numpy().squeeze())
        
        dvdt_0 = force / self.mass 
        
        v_step_half = 1/2 * dvdt_0 * dt 
        v = v + v_step_half
        
        # explicitly defined change in q
        q_step_full = v * dt
        q = q + q_step_full
        
        # gradient full at t + dt 
        force = self.separate_force(q, 0.001)
        
        dvdt_full = force / self.mass 
        
        v_step_full = v_step_half + 1/2 * dvdt_full * dt
        
        new_v = state[0] + v_step_full
        new_q = state[1] + q_step_full
        
        state = (new_v.detach(), new_q.detach())
        
        return state


    # def integrate_verlet_explicit(self, state, dt):
        
    #     def fce_force(q):
    #         q1 = q[0]
    #         q2 = q[1]

    #         x_A = q1 - 0
    #         x_B = q2 - q1

    #         force1 = -k*(torch.abs(x_A) - r0) * torch.sign(x_A) + k * (torch.abs(x_B) - r0) * torch.sign(x_B)
    #         force2 = - k * (torch.abs(x_B) - r0) * torch.sign(x_B)

    #         return torch.Tensor([force1, force2]).view(-1, 1).to(device)

    #     v,q = copy.deepcopy(state)
    #     assert q[1] > q[0], 'the two atoms collided'
        
    #     force = fce_force(q)
    #     # self.forces_log.append(force.cpu().numpy().squeeze())
        
    #     dvdt_0 = force / self.mass 
        
    #     v_step_half = 1/2 * dvdt_0 * dt 
    #     v = v + v_step_half
        
    #     # explicitly defined change in q
    #     q_step_full = v * dt
    #     q = q + q_step_full
        
    #     # gradient full at t + dt 
    #     force = fce_force(q)
        
    #     dvdt_full = force / self.mass 
        
    #     v_step_full = v_step_half + 1/2 * dvdt_full * dt
        
    #     new_v = state[0] + v_step_full
    #     new_q = state[1] + q_step_full
        
    #     state = (new_v.detach(), new_q.detach())
        
    #     return state
num_models = 5

dt = 0.1 # how is dt defined?

traj_lengths = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
num_samples = 50
data = []
start = time.perf_counter()
for l in traj_lengths:
    traj_length_data = []
    for m in range(1, num_models+1):
        func = torch.load(f'results/spring/{m}_model.pt').to(device)
        func_system = ODEFunc_NoEndFixed(func)
        
        states_data = []
        for n in range(num_samples):
            
            # axis = 0 => bond id; axis = 1 => (momentum, position)
            # get a random number from 0 to 1

            # position0 = 0.0
            # position1 = 1.0
            # position2 = 1.7
            # velocity0 = 0.0
            # velocity1 = 0.0
            # velocity2 = 0.0

            # state = torch.Tensor([[velocity0, position0], [velocity1, position1], [velocity2, position2]]).view(3, -1).to(device)
            state = torch.rand((3, 2)).to(device)
            state[:, 0] = 0
            state[0, 1] -= 1
            state[-1, 1] =+ 1

            states = [state.detach().cpu().numpy()]
            explicit_states = [state.detach().cpu().numpy()]

            state = torch.split(state, 1, dim=1)
            explicit_state = copy.deepcopy(state)


            with torch.no_grad():
                for i in range(l):
                    state = func_system.integrate_verlet(state, dt)
                    new_state = torch.cat(state, dim=1).detach().cpu().numpy().reshape(3, -1)
                    states.append(new_state)
                    
            states_data.append(np.array(states))
        
        traj_length_data.append(states_data)
    traj_length_data = np.array(traj_length_data)
    np.save(f'results/spring/divergence/{l}_traj_length_data.npy', traj_length_data)
