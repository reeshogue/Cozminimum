import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	def __init__(self, io_size, inchan):
		super(TargetActivation, self).__init__()
		self.inchan = inchan
		self.io_size = io_size
		self.conv = nn.Conv2d(inchan, inchan, 3)
		self.conv_backward = nn.Conv2d(inchan, inchan, 3)
		self.conv
		self.resize = lambda x: F.interpolate(x, self.io_size, mode='bicubic', align_corners=True)
	def forward(self, x):
		x = self.resize(x)
		conv = self.conv(x)
		y = self.resize(conv)
		return y

	def backward(self, y):
		y = self.resize(y)
		conv = self.conv_backward(y)
		x = self.resize(conv)
		return x
	def step(self, x, d_y):
		d_x = self.backward(d_y)
		dd_y = self.forward(d_x)

		loss_forward = torch.nn.functional.mse_loss(dd_y, d_y)
		self.optim_forward.zero_grad()
		loss_forward.backward()
		self.optim_forward.step()

		y_x = self.forward(x)
		x_y = self.backward(y_x)
		loss_backward = torch.nn.functional.mse_loss(x_y, x)
		self.optim_backward.zero_grad()
		loss_backward.backward()
		self.optim_backward.step()

class LossBlock(nn.Module):
	def __init__(self, io_size):
		super(LossBlock, self).__init__()
		self.io_size = io_size
		self.conv_backward = nn.Conv2d(inchan, inchan, 3)
		self.resize = lambda x: F.interpolate(x, self.io_size, mode='bicubic', align_corners=True)
		self.backward_optim = torch.optim.SGD(self.conv_backward.parameters(), lr=1e-5)
	def forward(self, loss):
		loss = torch.full_like(torch.zeros((1,3)+self.io_size))
		return loss

	def backward(self, y):
		y = self.resize(y)
		conv = self.conv_backward(y)
		x = self.resize(conv)
		return x

	def step(self, loss, desired_loss=torch.zeros((1,3)+self.io_size)):
		y = self.forward(loss)
		x = self.backward(y)
		x_ = self.backward(desired_loss)
				
		self.backward_optim.zero_grad()
		loss_y.backward()
		self.backward_optim.step()



