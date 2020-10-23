import torch
from net import *
import numpy as np
from collections import deque
from optim import Ranger
from optim_v2 import Optim
import random
import time

def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Agent:
    def __init__(self, size):
        self.actor = Net(size)
        self.actor_target = Net(size)
        self.rewarder = NextNet(size)
        self.world = WorldNet(size)
        self.q = QNet(size)
        self.q2 = QNet(size)
        self.q_target = QNet(size)

        self.actor_optim = Optim(self.actor.parameters(), lr=1e-3)
        self.rewarder_optim = Optim(self.rewarder.parameters(), lr=1e-3)
        self.q_optim = Optim(self.q.parameters(), lr=1e-3)
        self.q2_optim = Optim(self.q.parameters(), lr=1e-3)
        self.world_optim = Optim(self.world.parameters(), lr=1e-3)

        self.memory = deque(maxlen=2000)
        self.eps = 1.0
        self.eps_decay = 0.993
        self.eps_min = 0.0
        self.future_gamma = 0.911
        self.past_gamma = 0.997
    def act(self, state):
        if torch.rand(1) < self.eps:
            action = np.random.choice(7)
            action_probs = torch.zeros((1,7))
            action_probs[0][action] = 1
        else:
            with torch.no_grad():
                action_probs = self.actor(state)
                action_probs_sig = torch.softmax(action_probs, dim=-1)
                action_probs_numpy = action_probs_sig.numpy()[0]
                action = np.random.choice(7, p=action_probs_numpy)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        return action, action_probs

    def remember(self, prev_state, state, action, action_probs, reward, next_state):
        with torch.no_grad():
            reward = reward + torch.nn.functional.mse_loss(self.rewarder(state), next_state)
            self.memory.append((prev_state, state, action, action_probs, reward, next_state))


    def replay(self):
        time.sleep(2)
        while True:
            sampled = random.sample(self.memory, 1)
            for prev_state, state, action, action_probs, reward, next_state in sampled:
                self.world_optim.zero_grad()

                next_state_pred, reward_pred = self.world(state, action_probs)

                world_reward_loss = torch.nn.functional.mse_loss(reward_pred, reward)
                world_next_state_loss = torch.nn.functional.mse_loss(next_state_pred, next_state)
                world_loss = world_reward_loss + world_next_state_loss
                print("World Loss:", world_loss.item())

                world_loss.backward()
                self.world_optim.step()

                world_loss = world_loss.detach()
                reward = reward + world_loss

                self.actor_optim.zero_grad()
                loss = -self.q(state, self.actor(state))

                print("Q:", loss.item())
                loss.backward()
                self.actor_optim.step()
                
                self.q_optim.zero_grad()
                q = self.q(state, action_probs)
                prev_target = (self.past_gamma * self.q_target(prev_state, self.actor_target(prev_state)))
                next_target = (self.future_gamma * self.q_target(next_state, self.actor_target(next_state)))

                target = (prev_target + reward + next_target).detach()
                loss = torch.nn.functional.mse_loss(q, target)
                print("Q Loss:", loss.item())
                loss.backward()
                self.q_optim.step()

                self.q2_optim.zero_grad()
                q2 = self.q2(state, action_probs)
                target = target.detach()
                loss = torch.nn.functional.mse_loss(q2, target)
                print("Q2 Loss:", loss.item())
                loss.backward()
                self.q2_optim.step()

                self.rewarder_optim.zero_grad()
                next_state_pred = self.rewarder(state)
                loss = torch.nn.functional.mse_loss(next_state_pred, next_state)
                print("Rewarder Loss:", loss.item())
                loss.backward()
                self.rewarder_optim.step()

                with torch.no_grad():
                    if q2 < q:
                        soft_update(self.q_target, self.q2, tau=0.02)
                    else:
                        soft_update(self.q_target, self.q, tau=0.02)
                    soft_update(self.actor_target, self.actor, tau=0.02)
