import numpy as np

gamma = .99
gae_lambda = 0.95
use_gae = True
beta = .01
cliprange = 0.1
best_score = -np.inf
goal_score = 495.0

nenvs = 8
rollout_length = 200
minibatches = 10*8
# Calculate the batch_size
nbatch = nenvs * rollout_length
# nbatch = 128
optimization_epochs = 4