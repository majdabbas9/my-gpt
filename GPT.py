import torch
import torch.nn as nn
from TRANSFORMERS import TransformerBlock
from LayerNorm import LayerNorm
import tiktoken
from GPT_CONFIG import GPT_CONFIG_124M
from help_fucntions import generate_text_simple
class GPTModel(nn.Module):
   def __init__(self, cfg):
       super().__init__()
       self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
       self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
       self.drop_emb = nn.Dropout(cfg["drop_rate"])
       self.trf_blocks = nn.Sequential(
           *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) #A
       self.final_norm = LayerNorm(cfg["emb_dim"])          #B
       self.out_head = nn.Linear(
           cfg["emb_dim"], cfg["vocab_size"], bias=False
       )
   def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_emmbiding = self.tok_emb(in_idx)
        pos_emmbiding = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_emmbiding + pos_emmbiding
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

if __name__ == "__main__":
    main()
