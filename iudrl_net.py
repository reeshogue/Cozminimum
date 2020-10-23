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

class EnvNet(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBlock((32,32), 3, 3)
        self.conv2 = ConvBlock((16,16), 3, 3)
        self.conv3 = ConvBlock((32,32), 3, 3)
        self.conv4 = ConvBlock(size, 3, 3)
    def forward(self, state):
        return self.conv4(self.conv3(self.conv2(self.conv(state))))

class ActorNet(torch.nn.Module):
    def __init__(self, size, action_size):
        super().__init__()
        self.conv = ConvBlock((16,16), 6, 3)
        self.conv2 = ConvBlock((16,16), 3, 3)
        self.goal_linear = nn.Linear(16*16*3+1, 16*16*3)
        self.conv_goal = ConvBlock((16,16), 3, 3)
        self.output = nn.Linear(16*16*3, action_size)
    def forward(self, state, time, state_goal):
        state = torch.cat([state, torch.randn_like(state)], dim=1)
        time = time
        state_goal = self.conv_goal(state_goal)
        y = self.conv(state) + state_goal
        o_y = y
        y = torch.cat([torch.flatten(y, start_dim=1), time], dim=1)
        y = self.goal_linear(y)
        y = y.view_as(o_y)

        y = self.conv2(y)
        features = y
        o_y = y
        y = torch.cat([torch.flatten(y, start_dim=1), time], dim=1)
        y = self.goal_linear(y)
        y = y.view_as(o_y)

        y = torch.flatten(y, start_dim=1)
        y = self.output(y)
        y = torch.softmax(y, dim=1)
        return y, features

class DiscriminatorNet(nn.Module):
    def __init__(self, size, action_size):
        super().__init__()
        self.conv = ConvBlock(size, 6, 6)
        self.conv2 = ConvBlock((16,16), 6, 3)
        self.feature_conv = ConvBlock((16,16), 3, 3)
        self.action_linear = nn.Linear(action_size, 32)
        self.sg_linear = nn.Linear(16*16*3, 32)
        self.latent_linear = nn.Linear(64, 64)
        self.output_linear = nn.Linear(64, 1)

    def forward(self, action, state, goal, features):
        state_goal = torch.cat([state, goal], dim=1)
        sg = self.conv(state_goal)
        sg = self.conv2(sg)
        sg = sg + self.feature_conv(features)
        sg = torch.flatten(sg, start_dim=1)
        sg = self.sg_linear(sg)

        action = self.action_linear(action)

        cat = torch.cat([action, sg], dim=-1)
        latent = self.latent_linear(cat)
        output = self.output_linear(latent)
        output = torch.sigmoid(output)

        return output


class RewardNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv = ConvBlock(size, 3, 3)
        self.conv2 = ConvBlock(size, 3, 3)
        self.desired_state_conv = ConvBlock(size, 3, 3)
        self.time_conv = ConvBlock((16,16), 3, 3)
        self.time_linear = nn.Linear(16*16*3, 64)
        self.time_linear2 = nn.Linear(64, 1)
    def forward(self, state):
        base = self.conv(state)
        base = self.conv2(base)

        desired_state = self.desired_state_conv(base)

        time = self.time_conv(base)
        time = torch.flatten(time, start_dim=1)
        time = self.time_linear(time)
        time = self.time_linear2(time)

        return time, desired_state