"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Mélodie HANI DANIEL ZAKARIA
PhD Student
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.hani_daniel_zakaria@uca.fr
"""

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *
from mpi4py import MPI


# hidden_size=256

class DDPGagent:
    def __init__(self, cuda, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000, directory="./"):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.directory = directory
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.cuda = cuda

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
       
        
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
       
        
        sync_networks(self.actor)
        sync_networks(self.critic)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        if self.cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.cuda:
           state = state.cuda()
        self.actor.eval()
        with torch.no_grad():
             action = self.actor.forward(state).cpu().data.numpy()
        self.actor.train()
        return action.squeeze(0)
  
        
    def update(self, batch_size):
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states_batch)
        actions = torch.FloatTensor(actions_batch)
        rewards = torch.FloatTensor(rewards_batch)
        next_states = torch.FloatTensor(next_states_batch)
        dones = torch.from_numpy(np.vstack(dones_batch)).float()
        
        if self.cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
        
        # Critic loss        
        Qvals = self.critic(states, actions)
        with torch.no_grad():
             next_actions = self.actor_target(next_states)
             next_Q = self.critic_target(next_states, next_actions)
             Qprime = rewards + self.gamma * next_Q * (1.0 - dones)
             
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic(states, self.actor(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        sync_grads(self.actor)
       
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    
         
    def save(self):
        torch.save(self.actor.state_dict(), self.directory + 'actor.pth')
        torch.save(self.critic.state_dict(), self.directory + 'critic.pth')
        print("====================================")
        print("Model has been saved... for rank {}".format(self.rank))
        print("====================================")

    def load(self):
        if self.cuda:
              self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
              self.critic.load_state_dict(torch.load(self.directory + 'critic.pth'))
        else:
              self.actor.load_state_dict(torch.load(self.directory + 'actor.pth', map_location=torch.device('cpu')))
              self.critic.load_state_dict(torch.load(self.directory + 'critic.pth', map_location=torch.device('cpu')))
        print("====================================")
        print("model has been loaded... for rank {}".format(self.rank))
        print("====================================")
