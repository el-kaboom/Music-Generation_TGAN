# Generator model 
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel

class MusicGenerator(nn.Module):
    def __init__(self):
        super(MusicGenerator, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits
