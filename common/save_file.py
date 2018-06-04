#!/usr/bin/python
import h5py
import torch
import os
import numpy

model_dir = "model"
variable_dir = "var"

def entropy(p):
    p1 = np.exp(p) / np.sum(np.exp(p))
    return -sum(p1*np.log(p1))

def save_model(model, name):
    if not os.path.exists(model_dir): 
        os.makedir(model_dir)
    torch.save(model, model_dir + "/" + name)

def save_variable(state, name, qtrajectory, qentropy, action_size, *args):
    if not os.path.exists(variable_dir):
        os.makedir(variable_dir)
    
    state_size = list(qtrajectory[0][0].shape)
    state_size.insert(0, len(qtrajectory))
    state_trajectory = np.zeros(state_size)
    qvalue_trajectory = np.zeros((len(qtrajectory), action_size))
    qentropy = np.zeros(len(qtrajectory))

    for i in range(len(qtrajectory)):
        state_trajectory[i] = qtrajectory[i][0]
        qvalue_trajectory[i] = qtrajectory[i][1].data.cpu().numpy()[0]
        qentropy[i] = entropy(qvalue_trajectory[i])

    file_name = variable_dir + '/' + name

    with h5py.File(file_name, 'w') as hf:
        for key in args.keys():
            hf.create_dataset(key, data=args[key])
    
         

