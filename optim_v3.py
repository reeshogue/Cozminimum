from optim import Ranger
import torch
import torch.nn as nn
from torch.optim import Optimizer

class OptimActivation(nn.Module):
	def __init__(self):
		super(OptimActivation, self).__init__()
		self.params = nn.Parameter(torch.zeros(5))
	def forward(self, x):
		params_softmax = torch.softmax(self.params, dim=-1)

		x_swish = (torch.sigmoid(x) * x) * params_softmax[0]
		x_sine = torch.sin(x) * params_softmax[1]
		x_linear = x * params_softmax[2]
		x_sigmoid = (torch.sigmoid(x) * params_softmax[3])
		x_tanh = (torch.tanh(x)) * params_softmax[4]

		return x_swish + x_sine + x_linear

class OptimAttn(nn.Module):
	def __init__(self, size, out_size):
		super(OptimAttn, self).__init__()

		self.query_linear = nn.Linear(size, size, bias=False)
		self.key_linear = nn.Linear(size, size, bias=False)
		self.value_linear = nn.Linear(size, size, bias=False)

		self.activation = OptimActivation()
		self.activation2 = OptimActivation() 

		self.fc_linear = nn.Linear(size, size)
		self.norm = nn.LayerNorm((size,))

		self.fc_linear2 = nn.Linear(size, out_size)
		self.norm2 = nn.LayerNorm((out_size,))
	def forward(self, x):
		query = self.query_linear(x).unsqueeze(-1)
		key = self.key_linear(x).unsqueeze(-1)
		value = self.value_linear(x).unsqueeze(-1)
		
		attended = torch.matmul(query, key.transpose(1,2))
		attended = torch.softmax(torch.flatten(attended, start_dim=1), dim=-1).view_as(attended)
		valued = torch.matmul(attended, value)
		valued = valued.squeeze(-1)

		fc = self.activation(valued)

		fc = self.fc_linear(fc)
		fc = self.norm(fc)
		
		fc = self.activation2(valued)
		fc = self.fc_linear2(fc)
		fc = self.norm2(fc)
		
		return fc

class OptimLinear(torch.nn.Module):
	def __init__(self, inchan, outchan):
		super(OptimLinear, self).__init__()
		self.linear = nn.Linear(inchan, outchan)
		self.linear2 = nn.Linear(outchan, outchan)
		self.norm = nn.LayerNorm((outchan,))
		self.activation = OptimActivation()
	def forward(self, x):
		y = x
		y = self.activation(y)
		y = self.linear(y)
		y = self.linear2(y)
		y = self.norm(y)
		return y

class OptimNet(torch.nn.Module):
	def __init__(self, params):
		super(OptimNet, self).__init__()
		self.param_size = params.size()[-1]
		self.linear_h = OptimLinear(self.param_size * 2, self.param_size * 2)
		self.linear_a = OptimLinear(self.param_size * 2, self.param_size)
	def forward(self, x, sigmoid):
		y = self.linear_a(self.linear_h(x))
		if sigmoid:
			y = torch.sigmoid(y)
		return y

class OptimAgent(torch.nn.Module):
	def __init__(self, params):
		super(OptimAgent, self).__init__()
		self.actor = OptimNet(params)
		self.critic = OptimNet(params)
		self.actor_optim = Ranger(self.actor.parameters())
		self.critic_optim = Ranger(self.critic.parameters())

	def act(self, parameter, gradient, sigmoid=True):
		state = torch.cat([parameter, gradient], dim=-1)
		return self.actor(state, sigmoid).detach()

	def remember(self, loss, parameter, gradient, sigmoid=True):
		self.actor_optim.zero_grad()
		state = torch.cat([parameter, gradient], dim=-1).detach()
		action = self.actor(state, sigmoid=sigmoid)
		rewards = torch.mean(self.critic(torch.cat([action, gradient], dim=-1), False))
		rewards.backward()
		self.actor_optim.step()

		self.critic_optim.zero_grad()
		action = action.detach()
		gradient = gradient.detach()
		loss = loss.detach()

		state = torch.cat([action, gradient], dim=-1).detach()
		critique = torch.mean(self.critic(state, False))
		loss = torch.nn.functional.mse_loss(critique, loss)
		loss.backward()
		self.critic_optim.step()

class EMLYN(Optimizer):
	def __init__(self, params, lr=1e-2, betas=0.99):
		defaults = dict(lr=lr, betas=betas)
		super(EMLYN, self).__init__(params, defaults)
	def step(self, loss):
		for group in self.param_groups:
			for p in group['params']:

				grad_original_shape = p.grad.data.shape

				grad = p.grad.data.flatten().unsqueeze(0)
				parameters = p.data.flatten().unsqueeze(0)
				p_g = torch.cat([grad, parameters], dim=-1)

				state = self.state[p]
				if len(state) == 0:
					state['forgnet'] = OptimAgent(params=parameters)
					state['net'] = OptimAgent(params=parameters)
					state['neta'] = OptimAgent(params=parameters)

					state['momentum_buffer'] = torch.zeros_like(parameters)

				net = state['net']
				forgnet = state['forgnet']
				neta = state['neta']
				momentum_buffer = state['momentum_buffer']

				with torch.enable_grad():
					new_grad = net.act(parameters, grad, False)[0]
					new_forget = forgnet.act(parameters, grad)[0]
					new_neta = neta.act(parameters, grad)[0]

					net.remember(loss, parameters, grad, False)
					forgnet.remember(loss, parameters, grad)
					neta.remember(loss, parameters, grad)

				new_neta = group['betas']

				grad = (grad)
#				grad = grad * new_forget
				grad = grad * group['lr']

				momentum_buffer = (momentum_buffer * new_neta) + grad
				state['momentum_buffer'] = momentum_buffer

				grad = momentum_buffer + grad

				# grad *= new_forget
				grad *= group['lr']
				grad = grad.view(grad_original_shape)

				p.data = p.data - grad