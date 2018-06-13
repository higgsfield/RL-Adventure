import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import os
import gym
import random
import numpy as np

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from common.save_file import *

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
        if USE_CUDA else autograd.Variable(*args, **kwargs)


class CnnDQN(nn.Module):
  def __init__(self, input_shape, num_actions):
    super(CnnDQN, self).__init__()

    self.input_shape = input_shape
    self.num_actions = num_actions

    self.features = nn.Sequential(
        nn.Conv2d(input_shape[0] ,32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU()
    )

    self.fc = nn.Sequential(
        nn.Linear(self.feature_size(), 512),
        nn.ReLU(),
        nn.Linear(512, self.num_actions)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

  def feature_size(self):
    return self.features(autograd.Variable(
        torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

  def act(self, state):
    state   = Variable(torch.FloatTensor(np.float32(state)))
    q_value = self.forward(state[None])
    action  = q_value.max(1)[1].data[0]

    return action


env_id  = "DemonAttack-v0"
DIR     = "qEntropy"
EPISODE = 500
env     = make_atari(env_id)
env     = wrap_deepmind(env, clip_rewards=False)
env     = wrap_pytorch(env)

model_dir = os.path.join(DIR, "model")
model_name = DIR + "_" + env_id

model = load_model(model_dir, model_name)
if USE_CUDA:
  model.cuda()

scores = []
for i in range(EPISODE):
  episode_reward = 0
  state = env.reset()

  while True:
    env.render()
    action = model.act(state)
    state, reward, done, _ = env.step(action)
    episode_reward += reward

    if done:
      scores.append(episode_reward)
      print("Score : {}".format(episode_reward))
      break

env.close()

min_score = min(scores)
max_score = max(scores)
mean_score = np.mean(scores)
print("Min Score: {}, Max Score: {}, Mean Score: {}"\
    .format(min_score, max_score, mean_score))
