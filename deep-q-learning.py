#!/usr/bin/env python
# coding: utf-8

# In[25]:


import gym
from gym import wrappers
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import io
import base64
from IPython.display import HTML

#Set up env
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "./gym-results-pre", force=True)
env.reset()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
done = False

#Build NN
batch_size = 32
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
training_set = deque()
gamma = 0.95    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse',
              optimizer=Adam(lr=learning_rate))

# Decide wich action to take
def take_action(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    act_values = model.predict(state)
    return np.argmax(act_values[0])

# Train NN
def fit_actions():
    global epsilon
    minibatch = random.sample(training_set, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma *
                      np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Play
for e in range(1000):
    state = env.reset()
    state = np.array([state])
    for time in range(5000):
        # env.render()
        action = take_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.array([next_state])
        training_set.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            env.reset()
            print("episode: {}, score: {}, e: {:.2}"
                  .format(e, time, epsilon))
            break
        if len(training_set) > batch_size:
            fit_actions()


# In[26]:


video = io.open('./gym-results-pre/openaigym.video.%s.video000000.mp4' % env.file_infix, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''
    <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
.format(encoded.decode('ascii')))

