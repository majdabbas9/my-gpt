import torch.nn as nn
import torch
class MultiHeadAttention(nn.Module):
   def __init__(self, d_in, d_out, 
                context_length, dropout, num_heads, qkv_bias=False):
       super().__init__()
       assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
       self.d_out = d_out
       self.num_heads = num_heads
       self.head_dim = d_out // num_heads
       self.W_qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
       self.out_proj = nn.Linear(d_out, d_out)
       self.dropout = nn.Dropout(dropout)
       self.register_buffer(
           'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
       )
   def forward(self, x):
       batch_size, num_tokens, d_in = x.shape
       qkv = self.W_qkv(x)
       qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
       qkv = qkv.permute(2,0,3,1,4)
       queries, keys, values = qkv.unbind(0)
       attn_scores = queries @ keys.transpose(-2,-1)
       attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) 
       attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
       attn_weights = self.dropout(attn_weights)
       context_vec = (attn_weights @ values).transpose(1, 2)
       context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
       context_vec = self.out_proj(context_vec)                
       return context_vec

def main():
   inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your 
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with     
    [0.77, 0.25, 0.10], # one      
    [0.05, 0.80, 0.55]]) # step
   batch = torch.stack((inputs, inputs), dim=0)
   torch.manual_seed(123)
   batch_size, context_length, d_in = batch.shape
   d_out = 2
   mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
   context_vecs = mha(batch)
   print(context_vecs)


if __name__ == "__main__":
   main()        