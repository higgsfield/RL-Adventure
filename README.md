# DQN Adventure: from Zero to State of the Art


<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">

This is easy-to-follow step-by-step Deep Q Learning tutorial with clean readable code.

The deep reinforcement learning community has made several independent improvements to the DQN algorithm. This tutorial presents latest extensions to the DQN algorithm in the following order: 

  1. Playing Atari with Deep Reinforcement Learning [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb)
  2. Deep Reinforcement Learning with Double Q-learning [[arxiv]](https://arxiv.org/abs/1509.06461) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb)
  3. Dueling Network Architectures for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1511.06581) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/3.dueling%20dqn.ipynb)
  4. Prioritized Experience Replay [[arxiv]](https://arxiv.org/abs/1511.05952) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb)
  5. Noisy Networks for Exploration [[arxiv]](https://arxiv.org/abs/1706.10295) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb)
  6. A Distributional Perspective on Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1707.06887.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/6.categorical%20dqn.ipynb)
  7. Rainbow: Combining Improvements in Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1710.02298) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb)
  8. Distributional Reinforcement Learning with Quantile Regression [[arxiv]](https://arxiv.org/pdf/1710.10044.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/8.quantile%20regression%20dqn.ipynb)
  9. Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation  [[arxiv]](https://arxiv.org/abs/1604.06057) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/9.hierarchical%20dqn.ipynb)
  10. Neural Episodic Control [[arxiv]](https://arxiv.org/pdf/1703.01988.pdf) [[code]](#)

# Environments
First, I recommend to use small test problems to run experiments quickly. Then, you can continue on environments with large observation space. 

  - **CartPole** - classic RL environment can be solved on a single cpu
  - **Atari Pong** - the easiest atari environment, only takes ~ 1 million frames to converge, comparing with other atari games that take > 40 millions
  - **Atari others** - change hyperparameters, target network update frequency=10K, replay buffer size=1M

# If you get stuck… 
- Remember you are not stuck unless you have spent more than a week on a single algorithm. It is perfectly normal if you do not have all the required knowledge of mathematics and CS. For example, you will need knowledge of the fundamentals of measure theory and statistics, especially the [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) and [quantile regression](https://en.wikipedia.org/wiki/Quantile_regression). Statistical inference: [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling). Data structures: [Segment Tree](https://leetcode.com/tag/segment-tree/) and [K-dimensional Tree](https://en.wikipedia.org/wiki/K-d_tree).
- Carefully go through the paper. Try to see what is the problem the authors are solving. Understand a high-level idea of the approach, then read the code (skipping the proofs), and after go over the mathematical details and proofs.

# Best RL courses
- David Silver's course [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Berkeley deep RL [link](http://rll.berkeley.edu/deeprlcourse/)
- Practical RL [link](https://github.com/yandexdataschool/Practical_RL)
