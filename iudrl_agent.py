import torch
import numpy as np
import random

from collections import deque
from iudrl_net import ActorNet, EnvNet, DiscriminatorNet, RewardNet
from optim_v3 import EMLYN as Optim

class IUDRL_Agent:
	def __init__(self, size, action_size=7):
		self.actor = ActorNet(size, action_size)
		self.actor_optim = Optim(self.actor.parameters())
		self.loss_discriminator = DiscriminatorNet(size, action_size)
		self.loss_discriminator_optim = Optim(self.loss_discriminator.parameters())
		self.environment_model = EnvNet(size)
		self.env_optim = Optim(self.environment_model.parameters())

		self.reward = RewardNet(size)
		self.reward_optim = Optim(self.reward.parameters())

		self.memory = deque(maxlen=1000)

	def act(self, state):
		with torch.no_grad():
			action_probs, features = self.actor(state, *self.reward(state))
			action = np.random.choice(7, p=action_probs[0].numpy())

		return action, action_probs, features
	
	def remember(self, features, state, action, reward, next_state, timestep):
		with torch.no_grad():
			state = state
			action = action
			reward = reward
			next_state = next_state
			timestep = timestep

		timestep = timestep + torch.zeros((1,1))
		self.memory.append((features, state, action, reward, next_state, timestep))

	def replay(self):
		while True:
			latest_memory = random.sample(self.memory, 1)
			random_memory = random.sample(self.memory, 1)
			if latest_memory[-1][-1] < random_memory[-1][-1]:
				temp = latest_memory
				latest_memory = random_memory
				random_memory = temp


			for (rf, rs, ra, rr, rns, rt), (lf, ls, la, lr, lns, lt) in zip(random_memory, latest_memory):


				with torch.no_grad():
					dt = lt - rt
					dr = rr + lr
					ra = ra
					rs = rs
					ds = rs + ls
					rf = rf

				self.actor_optim.zero_grad()
				rs = rs * (torch.rand_like(rs) > .4)
				ra_pred, rf_pred = self.actor(rs, dt, ds)
				dt_pred, ds_pred = self.reward(rs)

				ra_pred_two, rf_pred_two = self.actor(rs, dt_pred, ds_pred)
				loss_one = self.loss_discriminator(ra_pred, ds, rs, rf_pred)
				loss_two = self.loss_discriminator(ra_pred_two, ds_pred, rs, rf_pred_two)
				loss = loss_one + loss_two


				print("Actor Loss:", loss.item())
				loss.backward()
				self.actor_optim.step()

				self.reward_optim.zero_grad()
				ra_pred, rf_pred = self.actor(rs, *self.reward(rs))
				loss = -self.loss_discriminator(ra_pred, ds, rs, rf_pred)
				print("Rewarder loss:", loss.item())
				loss.backward()
				self.reward_optim.step()

				self.loss_discriminator_optim.zero_grad()
				ra_pred, rf_pred = self.actor(rs, dt, ds)
				ones = self.loss_discriminator(ra_pred, ds, rs, rf_pred)
				zeros = self.loss_discriminator(ra, ds, rs, rf)
				ones_loss = torch.nn.functional.mse_loss(ones, torch.ones_like(ones))
				zeros_loss = torch.nn.functional.mse_loss(zeros, torch.zeros_like(zeros))
				loss = zeros_loss + ones_loss
				loss.backward()
				self.loss_discriminator_optim.step()

