import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = self.d_model // self.n_head

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)


        self.fc_out = nn.Linear(d_model, d_model)


    def forward(self, q, k, v):
        b_size = q.size(0)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = q.reshape(b_size, -1, self.n_head, self.head_dim)
        k = k.reshape(b_size, -1, self.n_head, self.head_dim)
        v = v.reshape(b_size, -1, self.n_head, self.head_dim)


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        sims = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_w = torch.softmax(sims, dim = -1)
        
        attn_o = torch.matmul(attn_w, v)
        attn_o = attn_o.transpose(1, 2)
        attn_o = attn_o.reshape(b_size, -1, self.d_model)
        attn_o = self.fc_out(attn_o)


        return attn_o