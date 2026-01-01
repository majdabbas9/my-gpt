import torch
import torch.nn as nn
from GPT_CONFIG import GPT_CONFIG_124M
class ExampleDeepNeuralNetwork(nn.Module):
   def __init__(self, layer_sizes, use_shortcut):
       super().__init__()
       self.use_shortcut = use_shortcut
       self.layers = nn.ModuleList([
           # Implement 5 layers
           nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
           nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
           nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
           nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
           nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
       ])
   def forward(self, x):
       for layer in self.layers:
           # Compute the output of the current layer
           layer_output = layer(x)
           # Check if shortcut can be applied
           if self.use_shortcut and x.shape == layer_output.shape:
               x = x + layer_output
           else:
               x = layer_output
       return x