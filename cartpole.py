import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp 
import pickle

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0) 
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c)) 
        return actor, critic 
    


def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) 
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env,worker_model)  
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards) 
        counter.value = counter.value + 1 

def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95): #clc= critic loss coefficient, we want actor to learn faster than the critic
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) 
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]): 
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        actor_loss = -1*logprobs * (Returns - values.detach()) 
        critic_loss = torch.pow(values - Returns,2) 
        loss = actor_loss.sum() + clc*critic_loss.sum() 
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)

def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float() 
    values, logprobs, rewards = [],[],[] 
    done = False
    j=0
    while (done == False): 
        j+=1
        policy, value = worker_model(state) 
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample() 
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done: 
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards




def multi_learner(n_workers=7):
    MasterNode = ActorCritic() 
    MasterNode.share_memory() 
    processes = [] #C
    params = {
        'epochs':1000,
        'n_workers':n_workers,
    }
    counter = mp.Value('i',0) 
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i,MasterNode,counter,params)) 
        p.start() 
        processes.append(p)
        print('worker number (i) :',  i)
    for p in processes: 
        p.join()
    for p in processes: 
        p.terminate()

    return MasterNode

def load_tester(model_name='model_scripted.pt'):
    model = torch.jit.load(model_name)

    env = gym.make("CartPole-v1")
    env.reset()

    for i in range(100):
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits,value = model(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        state2, reward, done, info = env.step(action.detach().numpy())
        if done:
            print("Lost after step number ", i)
            env.reset()
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()
    print('loop is over')

if __name__ == '__main__':    

    model = multi_learner(n_workers=7)
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('model_scripted.pt') # Save
    
    load_tester(model_name='model_scripted.pt')