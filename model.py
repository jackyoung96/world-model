import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import copy


def layer_init(layer, scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class FCNet(nn.Module):
    """Fully Connected Model."""
    def __init__(self, state_size, seed, hidden_layers, use_reset, act_fnc=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            hidden_layers (list): Size of hidden_layers
            act_fnc : Activation function
            use_reset (bool): Weights initialization
        """
        super(FCNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims = [state_size, ] + hidden_layers
        if use_reset:
            self.layers = nn.ModuleList([layer_init(nn.Linear(in_put, out_put)) for in_put, out_put in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList([nn.Linear(in_put, out_put) for in_put, out_put in zip(dims[:-1], dims[1:])])
        self.act_fuc = act_fnc 
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.act_fuc(layer(x))

        return x
    

class ConvNet(nn.Module):
    """Convolution Model for raw pixels."""
    def __init__(self, state_size, feature_dim, seed, use_reset=True, input_channel=4):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            feature_dim (int): Feature dimension
            seed (int): Random seed
            use_reset (bool): Weights initialization
            input_channel (int): Channel number of input
        """
        super(ConvNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.size1 = ((state_size-6)//2+1)  # 80x80x4 to 38x38x4
        self.size2 = ((self.size1-6)//4+1)  # 38x38x4 to 9x9x16
        self.size = self.size2**2*16
        self.feature_dim = feature_dim
        
        if use_reset:
            self.conv1 = layer_init(nn.Conv2d(input_channel, 4, kernel_size=6, stride=2))
            self.conv2 = layer_init(nn.Conv2d(4, 16, kernel_size=6, stride=4))
            self.fc3 = layer_init(nn.Linear(self.size, feature_dim))
        else:
            self.conv1 = nn.Conv2d(input_channel, 4, kernel_size=6, stride=2)
            self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
            self.fc3 = nn.Linear(self.size, feature_dim)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc3(x))
        return x


class PolicyDiscrete(nn.Module):
    """Actor Critic Model (shared weights)"""
    def __init__(self, state_size, action_size, seed, main_net):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            main_net (model): Common net for actor and critic 
        """
        super(PolicyDiscrete, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.main_net = main_net
        
        self.fc_actor = layer_init(nn.Linear(main_net.feature_dim, action_size), 1e-3)
        self.fc_critic = layer_init(nn.Linear(main_net.feature_dim, 1), 1e-3)
    
    def forward(self, state):
        x = self.main_net(state)
        pi_a = self.fc_actor(x)
        prob = F.softmax(pi_a, dim=1)
        v = self.fc_critic(x)
        return prob, v        
        
    def act(self, state, action=None):
        prob, v = self.forward(state)
        dist = Categorical(prob)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v.squeeze()}


class PolicyContinuous(nn.Module):
    """Actor Critic Model (shared weights)"""
    def __init__(self, state_size, action_size, seed, main_net1, main_net2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            main_net (model): Common net for actor and critic 
        """
        super(PolicyContinuous, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.main_net_actor = main_net1
        self.main_net_critic = main_net2

        self.fc_actor_mean = nn.Linear(main_net1.feature_dim, action_size)
        self.fc_actor_sigma = nn.Linear(main_net1.feature_dim, action_size)
        self.fc_critic = nn.Linear(main_net1.feature_dim, 1)

    def forward(self, state):
        x_a = self.main_net_actor(state)
        
        pi_mean = self.fc_actor_mean(x_a)
        pi_sigma = F.softplus(self.fc_actor_sigma(x_a))
        if self.main_net_critic is not None:
            x_c = self.main_net_critic(state)
            v = self.fc_critic(x_c)
        else:
            v = self.fc_critic(x_a)
        return (pi_mean,pi_sigma), v        
        
    def act(self, state, action=None):
        pi_a, v = self.forward(state)
        dist = Normal(*pi_a)
        if action is None:
            action = dist.sample()
        else:
            action = torch.unsqueeze(action, -1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v.squeeze()}