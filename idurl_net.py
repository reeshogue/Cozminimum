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

class ActorNet(torch.nn.Module):
    def __init__(self, size, action_size):
        super().__init__()
        self.conv = ConvBlock(size, 3, 3)
        self.conv2 = ConvBlock(size, 3, 3)
        self.conv3 = ConvBlock(size, 3, 3)
        self.resize = lambda x: F.interpolate(x, size=size)
        self.output = nn.Linear(size[0]*size[1]*3, action_size)
        self.output_determ = output_determ
    def forward(self, statwe):
        y = self.conv(state)
        y = self.resize(y)
        y = self.conv2(y)
        y = self.resize(y)
        y = self.conv3(y)
        y = self.resize(y)
        y = torch.flatten(y, start_dim=1)
        y = self.output(y)
        y = torch.softmax(y) 
        return y