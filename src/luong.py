import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()

        self.linear_in = nn.Linear(hidden_size, hidden_size,
                                   bias = False)
        
    def forward(self, hidden, encoder_outputs):
        hidden = self.linear_in(hidden)
        hidden = hidden.unsqueeze(2)
        attn_w = torch.bmm(encoder_outputs, hidden)
        attn_w = torch.softmax(attn_w.squeeze(2), dim = -1)
        context = torch.bmm(attn_w.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_w
