from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from transformers import MT5ForConditionalGeneration
import json

import utils

N_HIDDEN = 512

class CountModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CountModel):
            counts = [(k, list(v.items())) for k, v in obj.counts.items()]
            src_counts = [(k, list(v.items())) for k, v in obj.src_counts.items()]
            tgt_counts = [(k, list(v.items())) for k, v in obj.tgt_counts.items()]

            return {
                "_cls": "CountModel",
                "counts": counts,
                "totals": list(obj.totals.items()),
                "src_counts": src_counts,
                "src_totals": list(obj.src_totals.items()),
                "tgt_counts": tgt_counts,
                "tgt_totals": list(obj.tgt_totals.items()),
            }
        else:
            return super().default(obj)

def _tuplize(seq):
    if isinstance(seq, list):
        return tuple(_tuplize(s) for s in seq)
    return seq

def decode_count_model(obj):
    if "_cls" not in obj:
        return obj
    if obj["_cls"] == "CountModel":
        model = CountModel(None)
        model.counts = {k: dict(v) for k, v in _tuplize(obj["counts"])}
        model.totals = dict(_tuplize(obj["totals"]))
        model.src_counts = {k: dict(v) for k, v in _tuplize(obj["src_counts"])}
        model.src_totals = dict(_tuplize(obj["src_totals"]))
        model.tgt_counts = {k: dict(v) for k, v in _tuplize(obj["tgt_counts"])}
        model.tgt_totals = dict(_tuplize(obj["tgt_totals"]))
        return model
    assert False

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

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        assert n_layers == 1
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.attention_key = nn.Linear(hidden_size, hidden_size)
        self.attention_write = nn.Linear(hidden_size, hidden_size)

    def forward(self, src, tgt, state):
        h, c = state
        h = h.squeeze(0)
        c = c.squeeze(0)
        hiddens = []
        for i in range(tgt.shape[0]):
            h, c = self.cell(tgt[i], (h, c))
            key = self.attention_key(h)
            scores = (src * key.unsqueeze(0)).sum(dim=1, keepdim=True)
            weights = scores.softmax(dim=0)
            pooled = (src * weights).sum(dim=0)
            h = h + self.attention_write(pooled)
            hiddens.append(h)

        hiddens = torch.stack(hiddens)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        return hiddens, (h, c)


# Ordinary neural sequence model for comparison
class SequenceModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 32)
        self.enc = nn.LSTM(32, 256, 1)
        #self.dec = nn.LSTM(32, 256, 1)
        self.dec = LSTMWithAttention(32, 256, 1)
        self.pred = nn.Linear(256, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.PAD)
        self.vocab = vocab

    def sample(self, inp, max_len=20):
        inp_emb = self.emb(inp)
        inp_enc, state = self.enc(inp_emb)
        n_batch = inp.shape[1]
        out = torch.ones(1, n_batch).long() * self.vocab.START
        for i in range(max_len):
            out_emb = self.emb(out[-1:, :])
            hiddens, state = self.dec(inp_enc, out_emb, state)
            pred = self.pred(hiddens).squeeze(0)
            pred = (pred / .1).softmax(dim=1)
            samp = torch.multinomial(pred, num_samples=1)
            out = torch.cat([out, samp], dim=0)

        results = []
        for i in range(n_batch):
            seq = out[:, i].detach().cpu().numpy().tolist()
            if self.vocab.END in seq:
                seq = seq[:seq.index(self.vocab.END)+1]
            results.append(seq)
        return results


    def forward(self, inp, out):
        out_src = out[:-1, :]
        out_tgt = out[1:, :]

        inp_emb = self.emb(inp)
        out_emb = self.emb(out_src)

        inp_enc, state = self.enc(inp_emb)
        hiddens, _ = self.dec(inp_enc, out_emb, state)
        pred = self.pred(hiddens)

        pred = pred.view(-1, len(self.vocab))
        out_tgt = out_tgt.view(-1)
        loss = self.loss(pred, out_tgt)
        return loss

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
