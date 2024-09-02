import torch
import torch.nn as nn

import math



class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, attn_pdrop, resid_pdrop,
                 block_size):
        super(CausalSelfAttention, self).__init__()


        self.fc_attn = nn.Linear(n_embed, n_embed * 3)
        self.fc_out = nn.Linear(n_embed, n_embed)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)))

        self.n_head = n_head
        self.n_embed = n_embed
        self.block_size = block_size

    def forward(self, x):
        b_size, seq_len, dim = x.size()
        out = self.fc_attn(x)
        q, k, v = out.split(self.n_embed, -1)
        q = q.view(b_size, seq_len, self.n_head, dim // self.n_head)
        k = k.view(b_size, seq_len, self.n_head, dim // self.n_head)
        v = v.view(b_size, seq_len, self.n_head, dim // self.n_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        sims = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias.reshape(1, 1, self.block_size, self.block_size)[:, :, :seq_len, :seq_len] == 0
        sims = sims.masked_fill_(mask = mask, value = float('-inf'))
        sims = torch.softmax(sims, dim = -1)
        sims = self.attn_drop(sims)
        attn = torch.matmul(sims, v)
        attn = attn.transpose(1, 2)
        attn = attn.contiguous().view(b_size, seq_len, dim)
        attn = self.resid_drop(attn)
        attn = self.fc_out(attn)
        return attn
        

