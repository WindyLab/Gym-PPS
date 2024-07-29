import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__() 

        # self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.out_fn(self.fc4(h3))
        return out