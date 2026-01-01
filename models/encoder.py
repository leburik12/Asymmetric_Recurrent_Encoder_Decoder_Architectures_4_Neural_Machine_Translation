import torch
from torch.utils.data import Dataset, DataLoader

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.GRU(embed_size, hidden_size, num_layers)

    def forward(self, X, valid_len):
        X = self.embedding(X).permute(1, 0, 2)
        outputs, state = self.rnn(X)
        return outputs, state
