import math

import torch
from torch import nn

from .tokenizer import PAD


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=4096):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        rates = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(positions * rates)
        pe[:, 1::2] = torch.cos(positions * rates[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("sequence exceeds positional encoding capacity")
        return self.dropout(x + self.pe[:, :x.size(1)])


class TranslationTransformer(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.d = d
        self.src_embed = nn.Embedding(vocab, d, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(vocab, d, padding_idx=PAD)
        self.src_pos = PositionalEncoding(d, cfg["dropout"])
        self.tgt_pos = PositionalEncoding(d, cfg["dropout"])
        enc = nn.TransformerEncoderLayer(d, cfg["nhead"], cfg["dim_ff"], cfg["dropout"], batch_first=True)
        dec = nn.TransformerDecoderLayer(d, cfg["nhead"], cfg["dim_ff"], cfg["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, cfg["num_layers"], enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, cfg["num_layers"])
        self.project = nn.Linear(d, vocab)
        # Transformer stacks clone one prototype; independently initialize every
        # matrix after construction so layers do not start with identical weights.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.src_embed.weight[PAD].zero_()
            self.tgt_embed.weight[PAD].zero_()

    def encode(self, src):
        src_pad = src.eq(PAD)
        x = self.src_pos(self.src_embed(src) * math.sqrt(self.d))
        return self.encoder(x, src_key_padding_mask=src_pad), src_pad

    def _decode_hidden(self, memory, src_pad, tgt_in):
        tgt_pad = tgt_in.eq(PAD)
        t = tgt_in.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), 1)
        x = self.tgt_pos(self.tgt_embed(tgt_in) * math.sqrt(self.d))
        y = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_pad,
                         memory_key_padding_mask=src_pad)
        return y

    def decode_logits(self, memory, src_pad, tgt_in):
        return self.project(self._decode_hidden(memory, src_pad, tgt_in))

    def decode_step_logits(self, memory, src_pad, tgt_in):
        return self.project(self._decode_hidden(memory, src_pad, tgt_in)[:, -1:])

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode_logits(memory, src_pad, tgt_in)


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
