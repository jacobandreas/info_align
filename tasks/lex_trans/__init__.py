import os
import torch

DATA_PATH = f"{os.path.dirname(__file__)}/lex/es/train.txt"

class LexTransVocab:
    def __init__(self):
        vocab = {}
        for special_token in ["<pad>", "<skip1>", "<hole1>", "</s>"]:
            vocab[special_token] = len(vocab)

        with open(DATA_PATH) as reader:
            for line in reader:
                tr, en = line.split()
                for c in list(tr) + list(en):
                    if c not in vocab:
                        vocab[c] = len(vocab)

        self.SKIP1 = vocab["<skip1>"]
        self.HOLE1 = vocab["<hole1>"]
        self.START = self.HOLE1 # vocab["<start>"]
        self.SEP = vocab["</s>"]
        self.END = vocab["</s>"]
        self.PAD = vocab["<pad>"]

        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}

    def encode(self, seq):
        return [self.vocab[c] for c in seq]

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy().tolist()
        seq = [s for s in seq if s != self.PAD]
        return "".join(self.rev_vocab[c] for c in seq)

    def __len__(self):
        return len(self.vocab)

def load():
    vocab = LexTransVocab()
    data = []
    with open(DATA_PATH) as reader:
        for line in reader:
            tr, en = line.split()
            tr = vocab.encode(tr)
            en = vocab.encode(en)
            data.append((tr, en))
    return data, vocab
