#!/usr/bin/python
import torch
import os
import pickle

def save_model(model, model_dir, name):
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)
    file_name = model_dir + "/" + name + ".model"
    torch.save(model, file_name)

def save_variable(name, var_dir, var_dict):
    if not os.path.exists(var_dir):
        os.mkdir(var_dir)
    
    file_name = var_dir + '/' + name + ".pkl"

    with open(file_name, 'wb') as f:
        pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
