import torch
import math
from collections import deque

class Optim(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.992, 0.9), eps=1e-7, k=4, alpha=0.5):
		defaults = dict(lr=lr, 
						betas=betas, 
						eps=eps,
						buffer=[[None, None, None, None] for _ in range(10)],
						k=k,
						alpha=alpha)

		super(Optim, self).__init__(params, defaults)

		#Lookahead
		for group in self.param_groups:
			group['counter'] = 0
	
	def __setstate__(self, state):
		super(Optim, self).__setstate__(state)


	@torch.no_grad()
	def step(self):
		for group in self.param_groups:
			for p in group['params']:
				grad = p.grad.data
				state = self.state[p]

				if len(state) == 0:
					state['step'] = 0
					state['exp_avg'] = torch.zeros_like(p.data)
					state['exp_avg_sq'] = torch.zeros_like(p.data)
					state['previous_grad'] = torch.zeros_like(p.data)
					state['exp_variance'] = torch.zeros_like(p.data)
					state['end_gradient_mean'] = torch.zeros_like(p.data)
					state['end_gradient_var'] = torch.zeros_like(p.data)


				if state['step'] != 0:
					grad = grad - torch.sqrt(state['exp_variance']) / state['exp_avg']

				#Get previous exponential moving average.
				exp_avg, exp_avg_sq, previous_grad, exp_variance = state['exp_avg'], state['exp_avg_sq'], state['previous_grad'], state['exp_variance']
				
				#Get betas, lr, buffer and increase step.
				beta1, beta2, beta3 = group['betas']
				lr = group['lr']
				state['step'] += 1
				buffered = group['buffer'][int(state['step'] % 10)]
				
				exp_avg_prev = exp_avg
				exp_avg = torch.mul(exp_avg, beta1) + (1-beta1) * grad
				exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2) * (grad*grad)
				exp_variance = torch.mul(exp_variance, beta1) + (1-beta1) * (grad - exp_avg_prev) * (grad - exp_avg)
				exp_std = torch.sqrt(exp_variance)

				state['exp_avg'] = exp_avg
				state['exp_avg_sq'] = exp_avg_sq
				state['exp_variance'] = exp_variance



				#Diff grad calculations.
				diff_grad  = torch.abs(previous_grad - grad)
				dfc  = torch.div(1.0, (1.0 + torch.exp(-diff_grad)))
				state['previous_grad'] = grad
				exp_avg = exp_avg * dfc

				#Radam calculations.
				if state['step'] == buffered[0]:
					N_sma, step_size = buffered[1], buffered[2]
				else:
					buffered[0] = state['step']
					beta2_t = beta2 ** state['step']
					N_sma_max = 2/(1-beta2) - 1
					N_sma = N_sma_max - 2 * state['step'] * beta2_t
					buffered[1] = N_sma

					if N_sma >= 5:
						step_size = (lr * math.sqrt((1-beta2_t)
													* (N_sma - 4)
													/ (N_sma_max - 4)
													* (N_sma - 4)
													/ (N_sma)
													* (N_sma_max)
													/ (N_sma - 2)))
					else:
						step_size = lr / (1 - beta1 ** state['step'])
					buffered[2] = step_size

				if N_sma >= 5:
					denom = exp_avg_sq.sqrt() + group['eps']
					gradients = exp_avg / denom
					gradients, state = self.gradient_noise(gradients, state, beta1)
					step_in = gradients * -step_size
					p.data = p.data + step_in
				else:
					exp_avg, state = self.gradient_noise(exp_avg, state, beta1)
					p.data = p.data + (-step_size * exp_avg)

			#Lookahead.
			if group['counter'] == 0:
				for fast in group['params']:
					k, alpha = group['k'], group['alpha']
					param_state = self.state['fast']
					if 'slow_params' not in param_state:
						param_state['slow_params'] = torch.clone(fast.data).detach()
					slow = param_state['slow_params']
					fast.data.mul(alpha).add(slow, alpha=1.0-alpha)
					slow.data.copy_(fast)
			group['counter'] = (group['counter'] + 1) % group['k'] 

	def gradient_noise(self, gradients, state, beta):
		end_gradient_mean = state['end_gradient_mean']
		end_gradient_var = state['end_gradient_var']
		end_gradient_mean_prev = end_gradient_mean

		end_gradient_mean = torch.mul(end_gradient_mean, beta) + (1-beta) * (gradients)
		end_gradient_var = torch.mul(end_gradient_var, beta) + \
					   ((1-beta) * (gradients - end_gradient_mean_prev) * (gradients - end_gradient_mean))

		end_gradient_std = torch.sqrt(end_gradient_var)

		gradient_dist = torch.distributions.Normal(end_gradient_mean, end_gradient_std)
		gradients = gradient_dist.sample()
		state['end_gradient_var'] = end_gradient_var
		state['end_gradient_mean'] = end_gradient_mean
		return gradients, state
