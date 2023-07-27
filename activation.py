import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class SoftMax(nn.Module):

    def __init__(self, model, args):
        super().__init__()

        # embedding model
        self.embedding = model

        # classifier model
        self.classifier = nn.Linear(args.n_embed, args.n_keyword)

    def forward(self, x):
        x = self.embedding(x)
        x = self.classifier(x)

        return x