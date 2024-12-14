import os 
os.environ["SUPPRESS_MA_PROMPT"] = '1'

import numpy as np
from make_env import make_env

env = make_env('simple_tag')


# print('numer of agents', env.n)
# print('observation space', env.observation_space[0].shape)
# print('action space', env.action_space)
# print('n actions', env.action_space[0].n)

# observation = env.reset()
# print(observation)

no_op = np.array([0, 0.1, 0.2, 0.6, 0.1])
action = [no_op, no_op, no_op, no_op]
obs_, reward, done, info = env.step(action)
print(reward)
print(done)