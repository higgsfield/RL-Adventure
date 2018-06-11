import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd 

def entropy(p):
    p1 = np.exp(p) / np.sum(np.exp(p))
    return -sum(p1*np.log(p1))

def compute_q_entropy(q_values):
    q_entropy_traj = []

    for i in range(len(q_values)):
        q_entropy = []
        for j in range(len(q_values[i])):
            qval =autograd.Variable(torch.Tensor(q_values[i][j]))
            q_entropy.append(nn_entropy(qval))
        q_entropy_traj.append(q_entropy)
    
    return q_entropy_traj


def nn_entropy(p):
    p_softmax = nn.Softmax()(p).data.cpu().numpy()[0]
    return -np.sum(p_softmax*np.log(p_softmax))


def nn_log_entropy(p):
    p_softmax = nn.LogSoftmax()(p).data.cpu().numpy()[0]
    return -np.sum(p_softmax*np.log(p_softmax))
