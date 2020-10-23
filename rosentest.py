import torch
import math
from optim_v3 import EMLYN

def rosenbrock(tensor):
	x, y = tensor
	return (1-x) ** 2 + 1 * (y - x ** 2) ** 2


def rastrigin(tensor, A=10):
    return A + sum([(x**2 - (A/2) * torch.cos(2 * math.pi * x)) for x in tensor])
	# x, y = tensor

	# tau = 2 * 3.1415926535897932


def optimize_emlyn(optim):
	lr = 1e-3
	state = (2.0, 2.0)
	loc_min = (1, 1)
	momentum = 0.9
	x = torch.Tensor(state).requires_grad_(True)

	optim = optim([x], lr=lr, betas=momentum)
	for _ in range(400):
		optim.zero_grad()
		y = rastrigin(x)
		y.backward(retain_graph=True)
		optim.step(y)
	print(y.clone().detach().numpy())

def optimize_other(optim):
	lr = 1e-3
	state = (2.0, 2.0)
	loc_min = (1, 1)
	momentum = 0.9
	x = torch.Tensor(state).requires_grad_(True)

	optim = optim([x], lr=lr, momentum=momentum)
	for _ in range(800):
		optim.zero_grad()
		y = rastrigin(x)
		y.backward(retain_graph=True)
		optim.step()
	print(y.clone().detach().numpy())

optimize_emlyn(EMLYN)
optimize_other(torch.optim.SGD)