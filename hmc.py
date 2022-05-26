import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import time
from mpl_toolkits import mplot3d

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


def Evaluate_H(func, q, p, M_inv):
    #std= np.asarray([1.0,0.5])
    V_o,grad = func(q,p)
    H_o = V_o + 0.5*( np.matmul(p.reshape(1,-1),np.matmul(M_inv,p.reshape(-1,1)))) # potential + kinetic energy
    H_o = H_o.reshape(-1)
    return H_o

def Ham_dynamics(func, q, p, M, M_inv, steps, eps, store=False):
    
    if store:
        path = []
        energy = []
        
        path.append(np.concatenate((q,p)))
        energy.append(Evaluate_H(func,q,p,M_inv))
    
    
    
    for s in range(steps):
        
        V,grad = func(q,p)
        
        grad_old = copy.deepcopy(grad)
        q_old = copy.deepcopy(q)
        
        p += -(eps/2.0)*(grad)
        q += eps*p
        
        V,grad = func(q,p)
        
        
        p += -(eps/2.0)*(grad)
        
        if store:
            path.append(np.concatenate((q,p)))
            energy.append(Evaluate_H(func,q,p,M_inv))
#             print(grad_old, grad)
#             if not np.array_equal(np.sign(grad_old), np.sign(grad)):
#                 print('change of gradient sign')
#                 print('energy', energy[-2], energy[-1])
#                 if abs(energy[-1] - energy[-2])>0.1:
#                     print('great energy change')
#                     print('positions', q_old, q)



    if store:
        return (q,p),path,energy
    
    return (q,p)

def hmc(func, q0, num_samples, eps, steps, store=False):
    
    samples = []
    accept_rate = 0
    q = q0
    dim = q0.shape[0]
    
    if store == True:
        stored_vals = np.zeros((num_samples,steps+1,dim*2))
        energies = np.zeros((num_samples,steps+1))
        
    ind = 0
    M = np.identity(dim)
    M_inv = np.linalg.inv(M)
    while len(samples) < num_samples:
        
        mean = np.zeros((dim))
        cov = M
        p = np.random.multivariate_normal(mean, cov)
        
        q_temp = copy.deepcopy(q)
        p_temp = copy.deepcopy(p)
        
        (q_f,p_f),path,energy = Ham_dynamics(func,q_temp,p_temp,M,M_inv,steps,eps,store=True)
        stored_vals[ind,:,:] = np.asarray(path)
        energies[ind,:] = np.asarray(energy).reshape(-1)
        
        p_f *= -1
        
        
        V_o,grad = func(q,p)
        V_f,grad = func(q_f,p_f)
        H_o = V_o + 0.5*np.matmul(p.reshape(1,-1),np.matmul(M_inv,p.reshape(-1,1)))
        H_o = H_o.reshape(-1)
        H_f = V_f + 0.5*np.matmul(p_f.reshape(1,-1),np.matmul(M_inv,p_f.reshape(-1,1)))
        H_f = H_f.reshape(-1)
        
        acceptance = H_o - H_f
        
        val = np.log(np.random.rand())
        #print(acceptance)
        
        if val < acceptance:
            q = q_f
            accept_rate += 1
        
        samples.append(q)
        
        if len(samples)%250 == 0 and len(samples)< 10000:
            if len(samples) == 250:
                recent_samples = np.asarray(samples)
            else:
                recent_samples = np.asarray(samples[-250:])
            M_inv = np.cov(recent_samples.T)
            try:
                M = np.linalg.inv(M_inv)
            except:
                M = np.identity(dim)
                M_inv = np.linalg.inv(M)
    
            # TODO: fix the linalg so that it does not crash when singular matrix
#             print(M_inv)
#             print(ind)
        ind+=1
        
    acceptance = accept_rate/num_samples
        
    return samples,stored_vals,energies,acceptance

def run(potential, potential_fce):
    num_samples = [5000]
    init = np.random.randn(2)*2
    traj_length = 100
    if potential == '2d_shell':
        traj_step_size = 0.01
    else:
        traj_step_size = 0.1
    num_of_runs = 1

    for q in num_samples:
        for i in range(1, num_of_runs+1):
            start = time.perf_counter()
            samps,trajs,energies,acceptance = hmc(potential_fce, init, q, traj_step_size, traj_length, store=True)

            if not os.path.exists(f'results/{potential}/hmc/{q}'):
                os.makedirs(f'results/{potential}/hmc/{q}')

            with open(f'results/{potential}/hmc/{q}/{i}_hmc_samps.npy', 'wb') as f:
                np.save(f, samps)

            with open(f'results/{potential}/hmc/{q}/{i}_info.npy', 'wb') as f:
                np.save(f, np.array([time.perf_counter() - start, acceptance]))

            # just save the last one to get trajectories
            if i == (num_of_runs) and q == num_samples[-1]:
                with open(f'dataset/{potential}_test.npy', 'wb') as f:
                    np.save(f, trajs)

if __name__ == '__main__':
    for potential in ['wofe_quapp','10d_gaussian', '2d_shell']:
        if potential == 'wofe_quapp':
            potential_fce = Wofe_Quapp
        elif potential == '10d_gaussian':
            potential_fce = GaussianXD
        else:
            potential_fce = Shell2D

        run(potential, potential_fce) 
