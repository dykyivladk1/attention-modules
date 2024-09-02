import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()

        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.scale = torch.sqrt(torch.tensor(dim, dtype = torch.float32))

    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1) / self.scale)
        attn = torch.softmax(scores, dim = 1)
        output = torch.matmul(attn, v)
        return output