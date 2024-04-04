import sys

import matplotlib
from markdown_it.rules_core import inline

assert sys.version_info >= (3, 5)

import tensorflow as tf

assert tf.__version__ >= "2.0"

from tensorflow import keras
import numpy as np

import sklearn

assert sklearn.__version__ >= "0.20"

np.random.seed(42)
tf.random.set_seed(42)

TF_ENABLE_ONEDNN_OPTS = 0

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import gym

env = gym.make('CartPole-v1')

obs = env.reset()

print(obs)

print(env.action_space)
action = 0  # Accelarate right

# step() method executes given action and returns 5 values
obs, reward, done, trunc, info = env.step(action)

"""
obs - curr state of the game, after a step is performed or after it is reset. 
Observations are environment-dependent values. For cartpole game, it is a 1D NumPy array composed of 4 floats:
1. horizontal position of the cart
2. velocity of the cart
3. the angle of the pole
    a. 0 means pole is vertical
    b. positive(ie., >0) value means that the pole is slanting towards the right.
    c.negative(ie., <0) value means that the pole is slanting towards the left.
4. the angular velocity of the pole
reward - it is the reward the agent got for its previous step.
done - The sequence of steps between the moment the environment is reset until it is done is called an "episode". This will happen when the pole tilts too much or goes off the screen, or after the last episode (in this last case, you have won). done is a boolean which is True at the end of the episode, else done is False.
info - this environment-specific dictionary can provide some extra information that may be useful for debugging or for training.
"""
print(obs)
print(reward)
print(done)  # is game over
print(trunc)
print(info)

if (done):
    obs = env.reset()


# It says that CartPole has two values for action. 0 means left action, 1 means right action.
# print(env.observation_space)?

# help(env) #dictionary

# env.seed(42)

def basic_policy(obs1):
    #print(obs1)
    if obs1[1] == {}:
        angle = obs1[0][2]
    else:
        angle = obs1[2]
    #print(angle)
    if angle < 0:
        return 0
    else:
        return 1


totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(100):
        action = basic_policy(obs)
        obs, reward, done, trunc, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))