import torch.nn.functional as F
import torch

class RegressionHead1(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, embed_size):
        super(RegressionHead1, self).__init__()
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, 1)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits

class RegressionHead2(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, embed_size):
        super(RegressionHead2, self).__init__()
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp1 = torch.nn.Linear(embed_size, 256)
        self.mlp2 = torch.nn.Linear(256, 1)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        hidden1 = F.relu(self.mlp1(hidden_state))
        hidden2 = self.mlp2(hidden_state)
        return logits