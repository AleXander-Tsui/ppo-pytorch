import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.make_env import make_env                         # 把maddpg中的utils文件夹移到该代码目录下
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import pdb


# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

### 仿照maddpg中写的平行环境
def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

env_num = 100
agents_num = 4
episode_num = 30
epoch_num = 10000
if_share_policy = False
if_share_critic = False
if_mix = False           ### if if_mix == True, share_policy and share_critic == True
if_assign_id = False
env = make_parallel_env("simple_spread", env_num, seed, True)
num_state = env.observation_space[0].shape[0]  #agent*6
num_action = env.action_space[0].n
torch.manual_seed(seed)
#env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        if if_assign_id:
            self.fc1 = nn.Linear(num_state + agents_num, 100)
        else:
            self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state*agents_num, 100)              # 用的是shared_critic 所以输入的维度是num_state*agent_num
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

share_actor = Actor().cuda()
shared_critic = Critic().cuda()   # 给一个共用的critic

model_dir = Path('../exp')
if not model_dir.exists():
    curr_run = 'run1'
else:
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)

model_dir = '../exp/' + curr_run

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 20
    buffer_capacity = 1000
    batch_size = 2048

    def __init__(self, agent_i):          # 加一个编号agent_i，用来区别不同agent
        super(PPO, self).__init__()
        if if_share_critic:
            self.actor_net = share_actor
        else:
            self.actor_net = Actor().cuda()
        if if_share_critic:
            self.critic_net = shared_critic  # 多个agent用同一个critic
        else:
            self.critic_net = Critic().cuda()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter(model_dir)
        self.agent_i = agent_i

        self.id_vector = np.zeros(agents_num)
        self.id_vector[agent_i - 1] = 1

        self.env_buffer = []
        for i in range(env_num):
            self.env_buffer.append([])
        #print('buffer',self.env_buffer)


        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists(model_dir + '/net_param'):
            os.makedirs(model_dir + '/net_param')
            # os.makedirs('../param/img')

    def select_action(self, state):                 # 目前这个select_action只支持一个环境一个agent的状态输入[num_state, ]
        if if_assign_id:
            tmp = np.concatenate((state, self.id_vector))
            state = torch.from_numpy(tmp).float().unsqueeze(0).cuda()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).cuda()  # [1, num_state]  理论上可以改成支持多个环境一个agent的输入 即[env_num, num_state]
        with torch.no_grad():
            action_prob = self.actor_net(state)
        cg = Categorical(action_prob)
        action = cg.sample()
        return action.item(), action_prob[:,action.item()].item()


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, i_ep):               # 保存模型参数
        #import pdb; pdb.set_trace()
        torch.save(self.actor_net.state_dict(), model_dir + '/net_param/actor_net_%i_%i' % (self.agent_i, i_ep) + '.pkl')
        torch.save(self.critic_net.state_dict(), model_dir + '/net_param/critic_net_%i_%i' % (self.agent_i, i_ep) +'.pkl')

    def store_env_transition(self, env_id, transition):
        self.env_buffer[env_id].append(transition)
       
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update_with_parallel_env(self, i_ep, env_num):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).cuda()       
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).cuda()  
        reward = [t.reward for t in self.buffer]
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).cuda()
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).cuda()  
        Gt = []
        for i in range(env_num):
            R = self.critic_net(next_state[(i+1)*episode_num - 1])
            preward = reward[(i*episode_num):((i+1)*episode_num)]        # 这里的30应该对应着episode的长度，按顺序每隔一个episode求一次累计奖励 [待修改 把30换成episode_num]
            pGt = []
            for r in preward[::-1]:
                R = r + gamma * R
                pGt.insert(0, R)
            Gt = Gt + pGt
        Gt = torch.tensor(Gt, dtype=torch.float).cuda()  
        # pdb.set_trace()
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                #pdb.set_trace()
                if self.training_step % 1000 ==0:
                    #pdb.set_trace()
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                # pdb.set_trace()
                V = self.critic_net(state[index])    # critic网络的输入是所有agent的state
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                if if_assign_id:
                    batch_vec = torch.from_numpy(np.expand_dims(self.id_vector,0).repeat(len(index),axis=0)).float().cuda()
                    batch_state = torch.cat((state[index][:,((self.agent_i-1)*num_state):(self.agent_i*num_state)], batch_vec), 1)
                    action_prob = self.actor_net(batch_state).gather(1, action[index]) # new policy
                else:
                    action_prob = self.actor_net(state[index][:,((self.agent_i-1)*num_state):(self.agent_i*num_state)]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #self.writer.add_scalar('agent%i/loss/action_loss' % self.agent_i, action_loss, self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                # pdb.set_trace()
                value_loss = F.mse_loss(Gt_index, V)
                # pdb.set_trace()
                #self.writer.add_scalar('agent%i/loss/value_loss' % self.agent_i, value_loss, self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                self.writer.add_scalars('agent%i/losses' % self.agent_i,
                    {'vf_loss': value_loss,
                    'pol_loss': action_loss},
                    self.training_step)
                
                self.writer.add_scalars('agent%i/mean_episode_reward' % self.agent_i,
                    {'reward': torch.tensor(reward, dtype=torch.float).view(-1,1).mean()},
                    self.training_step)
                self.training_step += 1
        
        # pdb.set_trace()
        del self.buffer[:] # clear experience

class MAPPO(object):

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 20
    buffer_capacity = 1000
    batch_size = 2048

    def __init__(self,agents_num,env_num):
        #self.nagents = 4
        # self.agents = [PPO(agent_id)]
        self.agents = [PPO(i+1) for i in range(agents_num)]
        self.env_num = env_num
        self.training_step = 0

        self.actor_net = share_actor
        self.critic_net = shared_critic  # 多个agent用同一个critic

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)

        self.writer = SummaryWriter(model_dir)
        #import pdb; pdb.set_trace()

    def step(self,states):
        sum_actions = []
        sum_actions_prob = []
        sum_onehot_actions = []
        for i in range(len(states)):  #env_num
            one_env_actions = []
            one_env_actions_prob = []
            one_env_onehot_actions = []
            for a,state in zip(self.agents,states[i]):
                one_agent_action, one_agent_action_prob = a.select_action(state)
                # pdb.set_trace()
                one_agent_onehot_actions = np.zeros(num_action)
                one_agent_onehot_actions[one_agent_action] = 1
                one_env_actions.append(one_agent_action)
                one_env_actions_prob.append(one_agent_action_prob)
                one_env_onehot_actions.append(one_agent_onehot_actions)
                #import pdb; pdb.set_trace()
            sum_actions.append(one_env_actions)
            sum_actions_prob.append(one_env_actions_prob)
            sum_onehot_actions.append(one_env_onehot_actions)
        ## sum_actions env_num * num_agent * action
        #import pdb; pdb.set_trace()
        return sum_actions, sum_actions_prob, sum_onehot_actions

    def store_transition(self, state, actions, actions_prob, reward, next_state):
        for i in range(len(state)):
            share_state = state[i].flatten()
            share_next_state = next_state[i].flatten()
            for j, agent in enumerate(self.agents):
                trans = Transition(share_state, actions[i][j], actions_prob[i][j], reward[i][j], share_next_state)
                agent.store_env_transition(i,trans)

    def store_buff(self):
        for i in range(self.env_num):
            for agent in self.agents:
                agent.buffer = agent.buffer + agent.env_buffer[i]
                #import pdb; pdb.set_trace()
        for agent in self.agents:
            agent.env_buffer = []
        for i in range(self.env_num):
            for agent in self.agents:
                agent.env_buffer.append([])

    def update(self, i_epoch):
        for agent in self.agents:
            if len(agent.buffer) >= agent.batch_size:agent.update_with_parallel_env(i_epoch, self.env_num)

    def save(self, i_ep):
        for agent in self.agents:
            agent.save_param(i_ep)

    def mix_update(self, i_ep):
        mix_state_buffer = []
        mix_next_state_buffer = []
        mix_action_buffer = []
        mix_old_prob_buffer = []
        mix_Gt_buffer = []
        mix_agent_state_buffer = []
        for agent in self.agents:
            mix_state_buffer += [t.state for t in agent.buffer]
            next_state_buffer = [t.next_state for t in agent.buffer]
            mix_action_buffer += [t.action for t in agent.buffer]
            reward = [t.reward for t in agent.buffer]
            mix_old_prob_buffer += [t.a_log_prob for t in agent.buffer]
            #R = 0
            Gt = []
            for i in range(self.env_num):
                tmp_next_state = torch.tensor(next_state_buffer[(i+1)*episode_num - 1], dtype=torch.float).cuda()  
                #import pdb; pdb.set_trace()
                R = self.critic_net(tmp_next_state)
                preward = reward[(i*episode_num):((i+1)*episode_num)]     
                pGt = []
                for r in preward[::-1]:
                    R = r + gamma * R
                    pGt.insert(0, R)
                Gt = Gt + pGt
            mix_Gt_buffer += Gt  
            
            #### assign id
            if if_assign_id:
                mix_agent_state_buffer += [np.concatenate((t.state[(agent.agent_i-1)*num_state : (agent.agent_i*num_state)], agent.id_vector)) for t in agent.buffer]
            else:
            #### normal
                mix_agent_state_buffer += [t.state[(agent.agent_i-1)*num_state : (agent.agent_i*num_state)] for t in agent.buffer]

        share_state = torch.tensor(mix_state_buffer, dtype=torch.float).cuda()         
        action = torch.tensor(mix_action_buffer, dtype=torch.long).view(-1, 1).cuda()  
        #reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor(mix_old_prob_buffer, dtype=torch.float).view(-1, 1).cuda() 
        agent_state = torch.tensor(mix_agent_state_buffer, dtype=torch.float).cuda() 
        Gt = torch.tensor(mix_Gt_buffer, dtype=torch.float).cuda()  
        #import pdb; pdb.set_trace()
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(Gt))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                Gt_index = Gt[index].view(-1, 1)
                # pdb.set_trace()
                V = self.critic_net(share_state[index])    # critic网络的输入是所有agent的state
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                #import pdb; pdb.set_trace()

                ### actor网络的输入是当前agent的状态 所以需要截取一下
                action_prob = self.actor_net(agent_state[index]).gather(1, action[index]) # new policy

                #batch_vec = torch.from_numpy(np.expand_dims(self.id_vector,0).repeat(len(index),axis=0)).float().cuda()
                #batch_state = torch.cat((state[index][:,((self.agent_i-1)*num_state):(self.agent_i*num_state)], batch_vec), 1)
                #action_prob = self.actor_net(batch_state).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                #self.writer.add_scalar('agent%i/loss/action_loss' % self.agent_i, action_loss, self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                # pdb.set_trace()
                value_loss = F.mse_loss(Gt_index, V)
                # pdb.set_trace()
                #self.writer.add_scalar('agent%i/loss/value_loss' % self.agent_i, value_loss, self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                self.writer.add_scalars('losses',
                    {'vf_loss': value_loss,
                    'pol_loss': action_loss},
                    self.training_step)
                
                self.writer.add_scalars('mean_episode_reward',
                    {'reward': torch.tensor(reward, dtype=torch.float).view(-1,1).mean()},
                    self.training_step)
                self.training_step += 1
        
        # pdb.set_trace()
        for agent in self.agents:
            del agent.buffer[:] # clear experience

class Buffer(object):
    def __init__(self):
        self.buffer_capacity = 100000
        self.filled_i = 0 #未填充的第一个位置
    #def push(self, observations, actions, actions_prob, rewards, next_observations, dones):   
    
def main():
    print("agent num: ", agents_num)
    print("env num: ", env_num)
    print("episode_num: ", episode_num)
    print("epoch_num: ", epoch_num)
    MA = MAPPO(agents_num, env_num)
    with open(model_dir+"/log.txt", "w") as file:
        file.write("agent num: " + str(agents_num) +
                   "\nenv num: " + str(env_num) +
                   "\nepisode num: " + str(episode_num) +
                   "\nepoch num: " + str(epoch_num) +
                   "\nif_mix: " + str(if_mix) +
                   "\nif_share_policy: " + str(if_share_policy) +
                   "\nif_share_critic: " + str(if_share_critic) +
                   "\nif_assign_id: " + str(if_assign_id) +
                   "\nppo_update_time: " + str(MA.ppo_update_time) +
                   "\nbatch_size: " + str(MA.batch_size))

    
    for i_epoch in range(epoch_num):

        state = env.reset()   # env_num * agent_num * single_obs

        start = time.time()

        if render: env.render()
        for t in count():

            actions, actions_prob, onehot_actions = MA.step(state)          

            next_state, reward, done, _ = env.step(onehot_actions)

            MA.store_transition(state, actions, actions_prob, reward, next_state)

            state = next_state
            if t >= episode_num - 1: break
        
        MA.store_buff()
        if if_mix:
            MA.mix_update(i_epoch)
        else:
            MA.update(i_epoch)

        end = time.time()
        print('time_update',round(end-start,4))

        if ((i_epoch + 1) % 500 == 0): MA.save(i_epoch+1)


if __name__ == '__main__':
    start = time.time()
    main()
    #end = time.time()
    #print('time',end-start)
    print("end")
