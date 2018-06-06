import numpy as np

def entropy(p):
    p1 = np.exp(p) / np.sum(np.exp(p))
    return -sum(p1*np.log(p1))

def compute_q_entropy(q_values, tau):
    q_entropy_traj = []

    for i in range(len(q_values)):
        q_entropy = []
        for j in range(len(q_values[i])):
            q_entropy.append(entropy(q_values[i][j] / tau))
        q_entropy_traj.append(q_entropy)
    
    return q_entropy_traj

