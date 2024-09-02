import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attn_size):
        super(BahdanauAttention, self).__init__()

        self.attn = nn.Linear(hidden_size * 2, attn_size)
        self.v = nn.Parameter(torch.rand(attn_size))

    def forward(self, hidden_state, encoder_out):
        b_size, seq_len, hidden_size = encoder_out.size()

        hidden_ = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.cat((hidden_, encoder_out), dim=2)  

        energy = torch.tanh(self.attn(energy))  
        aligns = torch.matmul(energy, self.v)  

        attn_w = torch.softmax(aligns, dim=1)  

        context = torch.bmm(attn_w.unsqueeze(1), encoder_out)  
        context = context.squeeze(1)  
        return context, attn_w
