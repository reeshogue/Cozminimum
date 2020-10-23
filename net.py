import torch
import torch.nn as nn
import torch.nn.functional as F

class Megactivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = lambda x: torch.sigmoid(x)
        self.linear = lambda x: x
        self.swish = lambda x: torch.sigmoid(x) * x
        self.neg = lambda x: -x
        self.sin = lambda x: torch.sin(x)
        self.params = nn.Parameter(torch.zeros(5))
    def forward(self, x):
        params = torch.softmax(self.params, dim=-1)
        swish_y = self.swish(x) * params[0]
        sig_y = self.sigmoid(x) * params[1]
        linear_y = self.linear(x) * params[2]
        neg_y = self.neg(x) * params[3]
        sin_y = self.sin(x) * params[4]

        y = swish_y + sig_y + linear_y + neg_y + sin_y
        y = y / len(params)

        return y

class ConvBlock(nn.Module):
    def __init__(self, size, inchan, outchan, expansion=2):
        super().__init__()
        self.resize = lambda x: F.interpolate(x, size=size, mode='bicubic', align_corners=True)
        self.conv = nn.Conv2d(inchan, inchan*expansion, 3)
        self.norm = nn.LayerNorm((inchan*expansion,*size))
        self.activation = Megactivation()
        self.conv2 = nn.Conv2d(inchan*expansion, outchan, 3)
        self.norm2 = nn.LayerNorm((outchan,*size))
        self.activation2 = Megactivation()
        self.conv3 = nn.Conv2d(outchan, outchan, 3)
        self.norm3 = nn.LayerNorm((outchan,*size))
        self.activation3 = Megactivation()
        self.conv4 = nn.Conv2d(outchan, outchan, 3)
        self.norm4 = nn.LayerNorm((outchan,*size))
        self.activation4 = Megactivation()

        self.res = nn.Conv2d(inchan, outchan, 3)
    def forward(self, x):
        orig_x = x
        x = self.activation(x)
        x = self.conv(x)
        x = self.resize(x)
        x = self.norm(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.resize(x)
        x = self.norm2(x)

        orig_x = self.res(orig_x)
        orig_x = self.resize(orig_x)

        x = x + orig_x
        x = self.conv3(x)
        x = self.resize(x)
        x = self.norm3(x)
        x = self.activation3(x)

        x = self.conv4(x)
        x = self.resize(x)
        x = self.norm4(x)
        x = self.activation4(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, size, output_determ=True):
        super().__init__()
        self.conv = ConvBlock(size, 3, 3)
        self.conv2 = ConvBlock(size, 3, 3)
        self.conv3 = ConvBlock(size, 3, 3)
        self.resize = lambda x: F.interpolate(x, size=size)
        self.output = nn.Linear(size[0]*size[1]*3, 7)
        self.output_determ = output_determ
    def forward(self, state):
        y = self.conv(state)
        y = self.resize(y)
        y = self.conv2(y)
        y = self.resize(y)
        y = self.conv3(y)
        y = self.resize(y)
        if self.output_determ:
            y = torch.flatten(y, start_dim=1)
            y = self.output(y)
            y = torch.softmax(y) 
        return y

class NextNet(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBlock(size, 3, 3)
        self.conv2 = ConvBlock(size, 3, 3)
        self.conv3 = ConvBlock(size, 3, 3)
        self.resize = lambda x: F.interpolate(x, size=size)
        self.output = nn.Linear(size[0]*size[1]*3, 7)
    def forward(self, state):
        y = self.conv(state)
        y = self.resize(y)
        y = self.conv2(y)
        y = self.resize(y)
        y = self.conv3(y)
        y = self.resize(y)
        return y

class WorldNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBlock(size, 3, 3)
        self.linear = nn.Linear((size[0]*size[1]*3)+7, 32*32*1)
        self.conv2 = ConvBlock(size, 1, 3)
        self.conv3 = ConvBlock(size, 3, 3)

        self.next_state_baseline = ConvBlock(size, 3, 3)
        self.reward_baseline = ConvBlock(size, 3, 3)

        self.conv_ns = ConvBlock(size, 3, 3)
        self.linear_r = nn.Linear((size[0]*size[1]*3), 1)
    def forward(self, state, action):
        y = self.conv(state)
        orig_y = y
        y = torch.flatten(y, start_dim=1)
        y = torch.cat([action, y], dim=-1)
        y = self.linear(y)
        y = self.conv2(y.view(1,1,32,32)) + orig_y
        y = self.conv3(y)

        next_state_baseline = self.next_state_baseline(y)
        next_state = self.conv_ns(next_state_baseline)

        reward_baseline = self.reward_baseline(y)
        reward = self.linear_r(torch.flatten(y, start_dim=1))

        return next_state, reward

class QNet(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBlock(size, 3, 3)
        self.conv2 = ConvBlock(size, 3, 3)
        self.conv3 = ConvBlock(size, 3, 3)
        self.resize = lambda x: F.interpolate(x, size=size)
        self.output = nn.Linear(size[0]*size[1]*3, 16)
        self.output2 = nn.Linear(16+7, 16+7)
        self.output3 = nn.Linear(16+7, 1)
    def forward(self, state, action):
        y = self.conv(state)
        y = self.resize(y)
        y = self.conv2(y)
        y = self.resize(y)
        y = self.conv3(y)
        y = self.resize(y)
        y = torch.flatten(y, start_dim=1)
        y = torch.cat([self.output(y), action], dim=-1)
        y = self.output2(y)
        y = self.output3(y)
        return y
