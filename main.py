#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

import morph
import trans
import amr

from model import Model, PretrainedModel
from trainer import train
import info
import utils


TASK = "trans"
TRAIN = True


def visualize(model, vocab, data):
    model.eval()
    for x in range(200):
        src, tgt = data[x]
        print(vocab.decode(src))
        print(vocab.decode(tgt))

        choices = []
        for sj in (range(1, len(src))):
            for tj in range(1, len(tgt)):
                for invert in [False, True]:
                    s = info.score_split(src, tgt, 0, sj, len(src), 0, tj, len(tgt), invert, model, vocab)
                    choices.append((s, sj, tj, invert))

        best, sj, tj, invert = min(choices)
        s1s, s1e = 0, sj
        s2s, s2e = sj, len(src)
        t1s, t1e = 0, tj
        t2s, t2e = tj, len(tgt)
        if invert:
            t1s, t2s = t2s, t1s
            t1e, t2e = t2e, t1e
        print(vocab.decode(src[s1s:s1e]), " | ", vocab.decode(tgt[t1s:t1e]))
        print(vocab.decode(src[s2s:s2e]), " | ", vocab.decode(tgt[t2s:t2e]))
        print()


def main():
    random = np.random.RandomState(0)

    if TASK == "morph":
        data, vocab = morph.load()
    elif TASK == "trans":
        data, vocab = trans.load()
    elif TASK == "amr":
        data, vocab = amr.load()

    model_path = f"model_{TASK}.chk"

    model = PretrainedModel(vocab).cuda()
    #model = Model().cuda()
    if TRAIN:
        train(model, vocab, data, model_path, random)
    else:
        model.load_state_dict(torch.load(model_path))

    visualize(model, vocab, data)


if __name__ == "__main__":
    main()
