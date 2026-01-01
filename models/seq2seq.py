import torch
from torch.utils.data import Dataset, DataLoader

class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_valid_len):
        _, enc_state = self.encoder(src, src_valid_len)
        outputs, _ = self.decoder(tgt, enc_state)
        return outputs
