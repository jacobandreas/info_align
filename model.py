from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from transformers import MT5ForConditionalGeneration

import utils

N_HIDDEN = 512

# Estimates (conditional and unconditional) substring probabilities via counting.
# The `observe` functions increment the frequency of the corresponding event.
class CountModel:
    def __init__(self, vocab):
        self.counts = defaultdict(Counter)
        self.totals = Counter()

        self.src_counts = defaultdict(Counter)
        self.src_totals = Counter()
        self.tgt_counts = defaultdict(Counter)
        self.tgt_totals = Counter()
        self.vocab = vocab

    def observe(self, x, y):
        x = tuple(x)
        y = tuple(y)
        self.counts[x][y] += 1
        self.totals[x] += 1

    def observe_src(self, x, y, scale):
        x = tuple(x)
        y = tuple(y)
        self.src_counts[x][y] += 1. / scale
        self.src_totals[x] += 1. / scale

    def observe_tgt(self, x, y, scale):
        x = tuple(x)
        y = tuple(y)
        self.tgt_counts[x][y] += 1. / scale
        self.tgt_totals[x] += 1. / scale

    def h_src(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.src_counts[x][y]) - np.log(self.src_totals[x]))

    def h_tgt(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.tgt_counts[x][y]) - np.log(self.tgt_totals[x]))

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.counts[x][y]) - np.log(self.totals[x]))

# Estimates substring probabilities by fine-tuning a pre-trained model.
class PretrainedModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        self.vocab = vocab
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inp, out):
        # TODO double-check that this is necessary
        output = self.model(input_ids=inp, decoder_input_ids=out[:, :-1])
        logits = output.logits
        b, l, v = logits.shape
        return self.loss(logits.view(b*l, v), out[:, :-1].reshape(b*l)).view(b, l).sum(dim=1)

    def decode(self, inp):
        return self.model.generate(input_ids=inp, eos_token_id=self.vocab.END)

# Estimates substring probabilities by training a transformer from scratch.
class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), N_HIDDEN)
        self.pos_embedding = nn.Embedding(N_HIDDEN, N_HIDDEN)
        self.transformer = nn.Transformer(N_HIDDEN, batch_first=True)
        self.pred = nn.Linear(N_HIDDEN, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.PAD, reduction="none")

    def forward(self, inp, out):
        out_from = out[:, :-1]
        out_to = out[:, 1:]
        tgt_shape = out_to.shape

        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)
        out_pos = torch.arange(out_from.shape[1], device=out_from.device)[None, :]
        emb_out = self.embedding(out_from) + self.pos_embedding(out_pos)
        mask = nn.Transformer.generate_square_subsequent_mask(out_from.shape[1]).cuda()
        enc = self.transformer(emb_inp, emb_out, tgt_mask=mask)
        pred = self.pred(enc)

        pred = pred.reshape(-1, len(self.vocab))
        out_to = out_to.reshape(-1)

        loss = self.loss(pred, out_to).view(tgt_shape)
        loss = loss.sum(dim=1)
        return loss

    @torch.no_grad()
    def decode(self, inp, greedy=True):
        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)

        out = torch.tensor([[self.vocab.START]] * inp.shape[0]).cuda()

        for i in range(20):
            out_pos = torch.arange(out.shape[1], device=out.device)[None, :]
            emb_out = self.embedding(out) + self.pos_embedding(out_pos)
            mask = nn.Transformer.generate_square_subsequent_mask(out.shape[1]).cuda()
            enc = self.transformer(emb_inp, emb_out, tgt_mask=mask)
            pred = self.pred(enc)
            if greedy:
                choice = pred[:, -1:].argmax(dim=2)
            else:
                choice = torch.multinomial(torch.exp(pred[:, -1]), 1)
            out = torch.cat((out, choice), dim=1)

        results = []
        for row in out:
            row = row.cpu().numpy().tolist()
            if self.vocab.END in row:
                row = row[:row.index(self.vocab.END)+1]
            results.append(row)
        return results
