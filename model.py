import torch
from torch import nn
from transformers import MT5ForConditionalGeneration

import utils

N_HIDDEN = 512

class PretrainedModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        self.vocab = vocab

    def forward(self, inp, out):
        return self.model(input_ids=inp, labels=out).loss

    def decode(self, inp):
        return self.model.generate(input_ids=inp, eos_token_id=self.vocab.END)


class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), N_HIDDEN)
        self.pos_embedding = nn.Embedding(N_HIDDEN, N_HIDDEN)
        self.transformer = nn.Transformer(N_HIDDEN, batch_first=True)
        self.pred = nn.Linear(N_HIDDEN, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab[utils.PAD])

    def forward(self, inp, out):
        out_from = out[:, :-1]
        out_to = out[:, 1:]

        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)
        out_pos = torch.arange(out_from.shape[1], device=out_from.device)[None, :]
        emb_out = self.embedding(out_from) + self.pos_embedding(out_pos)
        mask = nn.Transformer.generate_square_subsequent_mask(out_from.shape[1]).cuda()
        enc = self.transformer(emb_inp, emb_out, tgt_mask=mask)
        pred = self.pred(enc)

        pred = pred.reshape(-1, len(self.vocab))
        out_to = out_to.reshape(-1)

        loss = self.loss(pred, out_to)
        return loss

    def decode(self, inp):
        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)

        out = torch.tensor([[self.vocab[utils.START]]] * inp.shape[0]).cuda()

        for i in range(20):
            out_pos = torch.arange(out.shape[1], device=out.device)[None, :]
            emb_out = self.embedding(out) + self.pos_embedding(out_pos)
            mask = nn.Transformer.generate_square_subsequent_mask(out.shape[1]).cuda()
            enc = self.transformer(emb_inp, emb_out, tgt_mask=mask)
            pred = self.pred(enc)
            choice = pred[:, -1:].argmax(dim=2)
            out = torch.cat((out, choice), dim=1)

        return out
