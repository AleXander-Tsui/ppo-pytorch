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
from utils.make_env import make_env                         # 把maddpg中的utils文件夹移到该代码目录下
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import pdb


# Parameters
gamma = 0.95
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

env_num = 60
env = make_parallel_env("simple_spread", env_num, seed, True)
#import pdb; pdb.set_trace()
#env = make_env("simple_spread", discrete_action=True)
num_state = env.observation_space[0].shape[0]  #agent*6
num_action = env.action_space[0].n
torch.manual_seed(seed)
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


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state*3, 100)              # 用的是shared_critic 所以输入的维度是num_state*agent_num
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

shared_critic = Critic()   # 给一个共用的critic
model_dir = '../exp/run4'

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 15
    buffer_capacity = 1000
    batch_size = 1024

    def __init__(self, agent_i):          # 加一个编号agent_i，用来区别不同agent
        super(PPO, self).__init__()
        self.actor_net = Actor().cuda()
        self.critic_net = Critic().cuda()   # 多个agent用同一个critic
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter(model_dir)
        self.agent_i = agent_i
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
        state = torch.from_numpy(state).float().unsqueeze(0).cuda()  # [1, num_state]  理论上可以改成支持多个环境一个agent的输入 即[env_num, num_state]
        #state = torch.from_numpy(state).float()
        with torch.no_grad():
            action_prob = self.actor_net(state)
        cg = Categorical(action_prob)
        #import pdb; pdb.set_trace()
        action = cg.sample()
        action = action.view(-1,1)
        #return action.item(), action_prob[:,action.item()].item()
        return action.cpu(), action_prob.gather(1,action).cpu()
        #return action.item(), action_prob

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):               # 保存模型参数
        #import pdb; pdb.set_trace()
        torch.save(self.actor_net.cpu().state_dict(), model_dir + '/net_param/actor_net_%i' % self.agent_i + '.pkl')
        torch.save(self.critic_net.cpu().state_dict(), model_dir + '/net_param/critic_net_%i' % self.agent_i +'.pkl')

    def store_env_transition(self, env_id, transition):
        self.env_buffer[env_id].append(transition)
       
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    #### update这个函数暂时用不上，它只能用于env_num=1时的参数更新
    def update(self, i_ep):           
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).cuda()
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).cuda()
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).cuda()

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).cuda()
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                #import pdb; pdb.set_trace()
                action_prob = self.actor_net(state[index][:,:num_state]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('agent%i/loss/action_loss' % self.agent_i, action_loss, self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('agent%i/loss/value_loss' % self.agent_i, value_loss, self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

    ### 针对env_num>1时的模型更新  主要需要注意的buffer中的数据排布 buffer中存的是单个agent的trans，长度为env_num*episode_num，比如3个环境、episode长度为30，
    ### 那么前30个trans是第一个env的，30~60是第二个env的，最后30个是第三个env的
    def update_with_parallel_env(self, i_ep, env_num):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).cuda()     
        #print('action',self.buffer[0].action)     
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).cuda()  
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).cuda()  
        #import pdb; pdb.set_trace()
        R = 0
        Gt = []
        #import pdb; pdb.set_trace()
        for i in range(env_num):
            preward = reward[(i*30):((i+1)*30)]        # 这里的30应该对应着episode的长度，按顺序每隔一个episode求一次累计奖励 [待修改 把30换成episode_num]
            pGt = []
            #import pdb; pdb.set_trace()
            for r in preward[::-1]:
                R = r + gamma * R
                pGt.insert(0, R)
            Gt = Gt + pGt
        Gt = torch.tensor(Gt, dtype=torch.float).cuda()  
        # pdb.set_trace()
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                # pdb.set_trace()
                V = self.critic_net(state[index])    # critic网络的输入是所有agent的state
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                #import pdb; pdb.set_trace()

                ### actor网络的输入是当前agent的状态 所以需要截取一下
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


        del self.buffer[:] # clear experience
class MAPPO(object):
    def __init__(self,agents):
        #self.nagents = 4
        # self.agents = [PPO(agent_id)]
        self.agents = agents
    def step(self,states):
        sum_actions = []
        sum_actions_prob = []
        for i in range(len(states)):  #env_num
            one_env_actions = []
            one_env_actions_prob = []
            for a,state in zip(self.agents,states[i]):
                one_agent_action, one_agent_action_prob = a.select_action(state)
                # pdb.set_trace()
                one_env_actions.append(one_agent_action)
                one_env_actions_prob.append(one_agent_action_prob)
            sum_actions.append(one_env_actions)
            sum_actions_prob.append(one_env_actions_prob)
        ## sum_actions env_num * num_agent * action
        return sum_actions, sum_actions_prob

class Buffer(object):
    def __init__(self):
        self.buffer_capacity = 100000
        self.filled_i = 0 #未填充的第一个位置
    #def push(self, observations, actions, actions_prob, rewards, next_observations, dones):


        
    
def main():
    ### 手动初始化agent，理想情况是用一个类初始化，然后agent都放在一个列表里 [待实现]
    agent1 = PPO(1)
    agent2 = PPO(2)
    agent3 = PPO(3)
    #agent4 = PPO(4)
    MA = MAPPO([agent1,agent2,agent3])
    '''
    for i_epoch in range(1500):
        state = env.reset()
        if render: env.render()
        
        for t in count():
            action1, action_prob1 = agent1.select_action(state[0])
            action2, action_prob2 = agent2.select_action(state[1])
            action3, action_prob3 = agent3.select_action(state[2])
            #import pdb; pdb.set_trace()
            one_hot_action1 = np.zeros(num_action)
            one_hot_action1[action1] = 1
            one_hot_action2 = np.zeros(num_action)
            one_hot_action2[action2] = 1
            one_hot_action3 = np.zeros(num_action)
            one_hot_action3[action3] = 1
            next_state, reward, done, _ = env.step([one_hot_action1, one_hot_action2, one_hot_action3])
            #next_state, reward, done = next_state[0], reward[0], done[0]
            #import pdb; pdb.set_trace()
            #next_state, reward, done, _ = env.step(action)
            trans1 = Transition(np.concatenate((state[0],state[1],state[2])), action1, action_prob1, reward[0], next_state[0])
            trans2 = Transition(np.concatenate((state[0],state[1],state[2])), action2, action_prob2, reward[1], next_state[1])
            trans3 = Transition(np.concatenate((state[0],state[1],state[2])), action3, action_prob3, reward[2], next_state[2])
            if render: env.render()
            agent1.store_transition(trans1)
            agent2.store_transition(trans2)
            agent3.store_transition(trans3)
            state = next_state

            if done[0] or t > 100:
                if len(agent1.buffer) >= agent1.batch_size:agent1.update(i_epoch)
                if len(agent2.buffer) >= agent2.batch_size:agent2.update(i_epoch)
                if len(agent3.buffer) >= agent3.batch_size:agent3.update(i_epoch)
                agent1.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                agent2.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                agent3.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
    agent1.save_param()
    agent2.save_param()
    agent3.save_param()
    '''
    
    ###### Sequential Env ######
    
    episode_num = 30
    for i_epoch in range(2000):
        #for j in range(env_num):     # 依次运行仿真环境
        state = env.reset()   # env_num * agent_num * single_obs
        start = time.time()
        if render: env.render()
        for t in count():
            ### 对于每一个agent挑选动作   理想情况下下面都要放在一个类中，一次性完成agent列表中所有agent的操作
            # action1, action_prob1 = agent1.select_action(state[0])
            # action2, action_prob2 = agent2.select_action(state[1])
            # action3, action_prob3 = agent3.select_action(state[2])
            # action4, action_prob4 = agent4.select_action(state[3])
            # start = time.time()
            # pdb.set_trace()
            actions, actions_prob = MA.step(state)  #actions env_num * num_agents * action_space
            # end = time.time()
            # print('time_MA_step',round(end-start,4))
            #print('actions',actions)
            ### 环境需要的是one-hot向量，手动生成one-hot   maybe有更好的实现方式 [待优化]
            one_hot_actions_nagents = []
            onehot_actions = []
            for i in range(env_num):
                onehot_actions.append([])
            for i in range(env_num):  
                for j in range(3):    #num_agents
                    one_hot_action = np.zeros(num_action)
                    one_hot_action[int(actions[i][j].item())] = 1
                    # pdb.set_trace()
                    one_hot_actions_nagents.append(one_hot_action)
                onehot_actions[i] = one_hot_actions_nagents
                one_hot_actions_nagents = []
            # one_hot_action1 = np.zeros(num_action)
            # one_hot_action1[action1] = 1
            # one_hot_action2 = np.zeros(num_action)
            # one_hot_action2[action2] = 1
            # one_hot_action3 = np.zeros(num_action)
            # one_hot_action3[action3] = 1
            # one_hot_action4 = np.zeros(num_action)
            # one_hot_action4[action4] = 1
            # old_actions = [one_hot_action1, one_hot_action2, one_hot_action3, one_hot_action4]
            # print('old_actions',old_actions)
            # time.sleep(10)
            ### 把one-hot拼起来作为最终的action与环境交互
            # start = time.time()
            # pdb.set_trace()
            next_state, reward, done, _ = env.step(onehot_actions)
            # end = time.time()
            # print('time_env_step',round(end-start,4))
            # 需要拼成每个env一个tra，并且按顺序
            ### 把得到的结果分别存成每个agent的trans  因为用的是shared_critic，所以state都存一样的，但更新actor网络时只用自己的state[i]，见update中的说明
            # start = time.time()
            for i in range(env_num):
                trans1 = Transition(np.concatenate((state[i][0],state[i][1],state[i][2])), actions[i][0].numpy()[0][0], actions_prob[i][0].numpy()[0][0], reward[i][0], next_state[i][0])
                trans2 = Transition(np.concatenate((state[i][0],state[i][1],state[i][2])), actions[i][1].numpy()[0][0], actions_prob[i][1].numpy()[0][0], reward[i][1], next_state[i][1])
                trans3 = Transition(np.concatenate((state[i][0],state[i][1],state[i][2])), actions[i][2].numpy()[0][0], actions_prob[i][2].numpy()[0][0], reward[i][2], next_state[i][2])
                #trans4 = Transition(np.concatenate((state[i][0],state[i][1],state[i][2],state[i][3])), actions[i][3].numpy()[0][0], actions_prob[i][3].numpy()[0][0], reward[i][3], next_state[i][3])
                if render: env.render()
                agent1.store_env_transition(i,trans1)
                agent2.store_env_transition(i,trans2)
                agent3.store_env_transition(i,trans3)
                #agent4.store_env_transition(i,trans4)
            state = next_state
            # end = time.time()
            # print('time_store',round(end-start,4))
            if t >= episode_num - 1: break
        for i in range(env_num):
            agent1.buffer = agent1.buffer + agent1.env_buffer[i]
            agent2.buffer = agent2.buffer + agent2.env_buffer[i]
            agent3.buffer = agent3.buffer + agent3.env_buffer[i]
            #agent4.buffer = agent4.buffer + agent4.env_buffer[i]
        # pdb.set_trace()
        agent1.env_buffer = []
        agent2.env_buffer = []
        agent3.env_buffer = []
        #agent4.env_buffer = []
        for i in range(env_num):
            agent1.env_buffer.append([])
            agent2.env_buffer.append([])
            agent3.env_buffer.append([])
            #agent4.env_buffer.append([])
        end = time.time()
        #print('time_buffer',round(end-start,4))
        start = time.time()
        ### 跑完所有环境后进行一次参数更新
        if len(agent1.buffer) >= agent1.batch_size:agent1.update_with_parallel_env(i_epoch, env_num)
        if len(agent2.buffer) >= agent2.batch_size:agent2.update_with_parallel_env(i_epoch, env_num)
        if len(agent3.buffer) >= agent3.batch_size:agent3.update_with_parallel_env(i_epoch, env_num)
        #if len(agent4.buffer) >= agent4.batch_size:agent4.update_with_parallel_env(i_epoch, env_num)
        end = time.time()
        #print('time_update',round(end-start,4))
        agent1.writer.add_scalar('liveTime/livestep', t, i_epoch)
        agent2.writer.add_scalar('liveTime/livestep', t, i_epoch)
        agent3.writer.add_scalar('liveTime/livestep', t, i_epoch)
        #agent4.writer.add_scalar('liveTime/livestep', t, i_epoch)
        # end2 = time.time()
        # print('time_update',end2-end)
    ### 保存参数
    agent1.save_param()
    agent2.save_param()
    agent3.save_param()
    #agent4.save_param()
    
    #######  Parallel Env  #######

    ### 一些关于parallel env的尝试，主要的困难点在于引入parallel env后，每次与环境交互，数据都会多一个维度，
    ### 之前对于单个环境来说，交互得到的数据是列表形式，列表长度是agent_num，按索引取出即是对应的数据
    ### 现在得到的数据是numpy格式，shape是[env_num, agent_num, state/action/reward_num]
    ### 而且之前在计算Gt时，我们用到的reward列表是按照env_num顺序分布的，现在需要并行也要做改动
    ### 最最理想的情况，修改后，我们能有一个MAPPO类，这个类能同时做多个agent多个env的并行更新和交互
    '''
    agent_num = 2
    agent = [PPO(i) for i in range(agent_num)]

    def parallel_select_action(agent_list, state, env):
        for i, agent in enumerate(agent_list):
            action, action_prob = agent.select_action(state[:, i, :])
            import pdb; pdb.set_trace()


    episode_num = 30
    for i_epoch in range(2000):
        state = env.reset()
        if render: env.render()
        parallel_select_action(agent, state, env)
    '''






if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('time',end-start)
    print("end")
