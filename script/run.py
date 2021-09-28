from numpy.core.numeric import _rollaxis_dispatcher
import gym
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from DrivingSchool.envs.DrivingSchool_env import DrivingSchoolEnv

class buffer:
    def __init__(self, obs_dim, act_dim, size=30000, device='cpu'):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.r_buf = np.zeros(size, dtype=np.float32)
        self.d_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = size
        self.device = device
        self.is_full = False
    
    def reset(self):
        self.obs_buf *= 0
        self.act_buf *= 0
        self.r_buf *= 0
        self.d_buf *= 0
        self.ptr = 0
        self.is_full = False

    def store(self, obs, act, r, d):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.r_buf[self.ptr] = r
        self.d_buf[self.ptr] = d
        self.ptr += 1
        if self.ptr == self.size:
            self.is_full = True
        else:
            self.is_full = False
        assert self.ptr <= self.size
    
    def get(self):
        epoch_data = dict(obs=self.obs_buf, act=self.act_buf, r=self.r_buf, d=self.d_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in epoch_data.items()}


class policynet(nn.Module):
    def __init__(self, obs_dim, act_dim, activation, output_activation=nn.Identity):
        super().__init__()
        self.obs_dim = obs_dim
        self.l1 = nn.Linear(self.obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, act_dim)
        self.activation = activation
        self.output_activation = output_activation
    def forward(self, obs):
        if len(obs.size())==1:
            obs = obs.unsqueeze(0)

        activation = self.activation()
        output_activation = self.output_activation()
        h = activation(self.l1(obs))
        h = activation(self.l2(h))
        h = output_activation(self.l3(h))
        out = torch.squeeze(h, 0)
        return out


def update(data, net, optimizer, steps=200):
    obs = data['obs']
    act = data['act']
    for _ in range(steps):
        loss = ((net(obs)-act)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = DrivingSchoolEnv()
    device = 'cpu'
    net = policynet(env.observation_space.shape[0], env.action_space.shape[0], nn.Tanh).to(device=device)
    buf = buffer(env.observation_space.shape[0], env.action_space.shape[0], device=device)
    optimizer = Adam(net.parameters(), lr=0.01)
    o = env.reset()
    o_torch = torch.as_tensor(o, dtype=torch.float32, device=device)
    total_r = 0
    ep_r = []
    # print(o)
    while True:
        a = net(o_torch).cpu().detach().numpy()
        noise = np.clip(np.random.normal(size=env.action_space.shape[0]), -1., 1.)
        a += noise

        new_o, r, d, info = env.step(a)
        total_r += r

        o_torch = torch.as_tensor(new_o, dtype=torch.float32, device=device)
        buf.store(o, a, r, d)
        if buf.is_full:
            data = buf.get()
            update(data, net, optimizer)
            buf.reset()
            ave_ep_r = np.mean(ep_r)
            ep_r = []
            print(ave_ep_r)
        o = new_o
        # env.render()
        if d:
            ep_r.append(info["total_distance"])
            total_r = 0
            o = env.reset()
            o_torch = torch.as_tensor(o, dtype=torch.float32, device=device)


if __name__ == '__main__':
    main()