from xml.etree.ElementTree import TreeBuilder
from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import dim
from sympy.printing.pytorch import torch
import torch
import tiktoken
from GPT_CONFIG import GPT_CONFIG_124M
from help_fucntions import generate_text_simple,text_to_token_ids,token_ids_to_text,tokenizer,create_dataloader_v1,calc_loss_loader,train_model_simple
from GPT import GPTModel
model = GPTModel(GPT_CONFIG_124M)
torch.manual_seed(123)
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
   text_data = file.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
torch.manual_seed(123)

train_loader = create_dataloader_v1(train_data,
   batch_size=2,
   max_length=GPT_CONFIG_124M["context_length"],
   stride=GPT_CONFIG_124M["context_length"],
   drop_last=True,
   shuffle=True,
   num_workers=0
   )
val_loader = create_dataloader_v1(
   val_data,
   batch_size=2,
   max_length=GPT_CONFIG_124M["context_length"],
   stride=GPT_CONFIG_124M["context_length"],
   drop_last=False,
   shuffle=False,
   num_workers=0
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #A
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
   model, train_loader, val_loader, optimizer, device,
   num_epochs=num_epochs, eval_freq=5, eval_iter=1,
   start_context="Every effort moves you", tokenizer=tokenizer
)