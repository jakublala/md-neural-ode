from diffmd.solvers import odeint_adjoint
from diffmd.utils import compute_grad

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f'Using {device} device')

from training import train_model, get_data, RunningAverageMeter, get_batch_mod, get_first_batch

def test_model(func, testing_trajs, dt):    
    loss_meter = RunningAverageMeter()
    batch_size = testing_trajs.shape[0]
    sample_length = testing_trajs.shape[1]
    batch_t, batch_y0, batch_y = get_first_batch(testing_trajs, batch_size, sample_length, dt)
        
    # convert momentum to velocity (v = p / mu)
    batch_y0 = (batch_y0[0] / func.mass, batch_y0[1])
    pred_y = odeint_adjoint(func, batch_y0, batch_t, method='NVE')

    # convert velocity to momentum (p = mu * v)
    pred_y = (pred_y[0] * func.mass, pred_y[1])
    pred_y = torch.cat(pred_y, dim=2)
    
    loss = torch.mean(torch.abs(pred_y[:, :, 0] - batch_y[:, :, 0]))
    loss_meter.update(loss.item())
    
    return loss_meter

def evaluate_model(sample_length, batch_size, learning_rate, nn_depth, nn_width, activation_function, scheduling_factor, scheduling_freq):
    niters = 5000
    dt = 0.1
    t0 = time.perf_counter()

    training_trajs, testing_trajs = get_data('wofe_quapp', 1.0)
    model, train_loss = train_model(niters, training_trajs, dt, sample_length, batch_size, learning_rate, scheduling_factor, scheduling_freq, nn_depth, nn_width)
    training_time = time.perf_counter() - t0

    try:
        test_loss = test_model(model, testing_trajs, dt)
    except:
        test_loss = None

    training_and_testing_time = time.perf_counter() - t0
    
    return train_loss, test_loss, training_time, training_and_testing_time


import sigopt

def run_and_track_in_sigopt():

    #   sigopt.log_dataset(DATASET_NAME)
    #   sigopt.log_metadata(key="Dataset Source", value=DATASET_SRC)
    #   sigopt.log_metadata(key="Feature Eng Pipeline Name", value=FEATURE_ENG_PIPELINE_NAME)
    #   sigopt.log_metadata(
    #     key="Dataset Rows", value=features.shape[0]
    #   )  # assumes features X are like a numpy array with shape
    #   sigopt.log_metadata(key="Dataset Columns", value=features.shape[1])
    #   sigopt.log_metadata(key="Execution Environment", value="Colab Notebook")
    
    # learning_rates = [10**i for i in range(-5, 1)]
    # sigopt.params.setdefaults(
    # sample_length=np.random.randint(low=3, high=50),
    # batch_size=np.random.randint(low=10, high=1000),
    # learning_rate=np.random.choice(learning_rates),
    # nn_depth=np.random.randint(low=1, high=5),
    # nn_width=np.random.randint(low=2, high=50),
    # # activation_function=,  
    # )

    args = dict(
    sample_length=20,
    batch_size=800,
    learning_rate=sigopt.params.learning_rate,
    nn_depth=2,
    nn_width=50,
    activation_function=None,
    scheduling_factor=1000,
    scheduling_freq=0.6,
    )

    sigopt.log_model('Wolfe Quapp')
    
    
    train_loss, test_loss, training_time, training_and_validation_time = evaluate_model(**args)

    running_avg_train_loss = train_loss.avg
    # running_avg_test_loss = test_loss.avg
    
    sigopt.log_metric(name="train_loss", value=running_avg_train_loss)
    # sigopt.log_metric(name="test_loss", value=running_avg_test_loss)
    sigopt.log_metric(name="training time (s)", value=training_time)
    sigopt.log_metric(name="training and validation time (s)", value=training_and_validation_time)

run_and_track_in_sigopt()