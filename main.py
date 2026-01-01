import torch
import tiktoken
from GPT_CONFIG import GPT_CONFIG_124M
from help_fucntions import generate_text_simple,text_to_token_ids,token_ids_to_text,tokenizer
from GPT import GPTModel
model = GPTModel(GPT_CONFIG_124M)
start_context = "Every effort moves you"
token_ids = generate_text_simple(
   model=model,
   idx=text_to_token_ids(start_context, tokenizer),
   max_new_tokens=10,
   context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))