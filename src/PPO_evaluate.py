import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from torch.autograd import Variable
import imageio

# Parameters
gamma = 0.95
render = False
seed = 1
log_interval = 10

env = make_env("simple_spread", discrete_action=True)
num_state = env.observation_space[0].shape[0]
num_action = env.action_space[0].n
#torch.manual_seed(seed)
#env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

if __name__ == '__main__':
    ### 手动加agent
    agent1 = Actor()
    agent2 = Actor()
    agent3 = Actor()
    agent4 = Actor()
    agent1.load_state_dict(torch.load('../param/net_param/actor_net_11582342728.pkl'))
    agent2.load_state_dict(torch.load('../param/net_param/actor_net_21582342742.pkl'))
    agent3.load_state_dict(torch.load('../param/net_param/actor_net_31582342758.pkl'))
    agent4.load_state_dict(torch.load('../param/net_param/actor_net_41582342780.pkl'))
    if not os.path.exists('../gifs'):
        os.makedirs('../gifs')
    n_episodes = 10
    save_gifs = True
    episode_length = 30
    nagents = 1
    ifi = 1 / 30
    gif_path = '../gifs'
    for ep_i in range(n_episodes):
        print("Episode %i of %i" % (ep_i + 1, n_episodes))
        obs = env.reset()
        if save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')
        for t_i in range(episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            state = torch.tensor([state for state in obs], dtype=torch.float)
            # get actions as torch Variables
            #import pdb; pdb.set_trace()
            ### 手动拼agent的state
            torch_actions = [agent1(state[0].view(-1, num_state)), agent2(state[1].view(-1, num_state)), agent3(state[2].view(-1, num_state)), agent4(state[3].view(-1, num_state))]

            # convert actions to numpy arrays
            prob_actions = [ac.data.numpy().flatten() for ac in torch_actions]
            #import pdb; pdb.set_trace()
            ### 手动生成one-hot向量
            actions = []
            for a in prob_actions:
                index = np.argmax(a)
                ac = np.zeros(num_action)
                ac[index] = 1
                actions.append(ac)

            #import pdb; pdb.set_trace()
            obs, rewards, dones, infos = env.step(actions)
            #import pdb; pdb.set_trace()
            if save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
        if save_gifs:
            gif_num = 0
            while os.path.exists('../gifs/%i_%i.gif' % (gif_num, ep_i)):
                gif_num += 1
            imageio.mimsave('../gifs/%i_%i.gif' % (gif_num, ep_i),
                            frames, duration=ifi)
