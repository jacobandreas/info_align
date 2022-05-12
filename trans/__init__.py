from collections import Counter
import itertools as it
import os
import re
from transformers import MT5Tokenizer

import utils

N = 50000
PATH_X = f"{os.path.dirname(__file__)}/europarl-v7.es-en.en"
PATH_Y = f"{os.path.dirname(__file__)}/europarl-v7.es-en.es"

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

class TransVocab:
    def __init__(self):
        self.SKIP1 = tokenizer.convert_tokens_to_ids("<extra_id_1>")
        self.HOLE1 = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        #self.START = tokenizer("<extra_id_0>", add_special_tokens=False).input_ids 
        self.START = self.HOLE1
        # FIXME(ekin): I think it will better to make this "<extra_id_0>" as well, T5 is trained for output="<extra_id_0> hole value <extra_id_1> second hole's value ...."
        # jda if we only ever need to predict on one side, then we can make this
        # much more t5-like
        #self.SEP = tokenizer("", add_special_tokens=False).input_ids
        self.SEP = tokenizer.eos_token_id
        self.END = tokenizer.eos_token_id
        self.PAD = tokenizer.pad_token_id

    def decode(self, seq):
        return tokenizer.decode(seq).replace(tokenizer.pad_token, "")

def load():
    def tokenize(line):
        tokenized = tokenizer(line)
        t = tokenized["input_ids"]
        assert t[-1] == tokenizer.eos_token_id
        return t[:-1]

    def read_from(path):
        with open(path) as reader:
            for line in it.islice(reader, N):
                yield tokenize(line)

    x = []
    for tokens in read_from(PATH_X):
        x.append(tokens)

    y = []
    for tokens in read_from(PATH_Y):
        y.append(tokens)

    pairs = [(x_, y_) for x_, y_ in zip(x, y) if len(x_) > 0 and len(y_) > 0]

    #return pairs[:1] * 1000, TransVocab()
    return pairs, TransVocab()
