import torch.nn.functional as F
import torch

class RegressionHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, embed_size):
        super(RegressionHead, self).__init__()
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, 1)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits