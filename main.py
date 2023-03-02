#!/usr/bin/env python3

import numpy as np
import torch
import pickle
import bz2
import json

from model import Model, PretrainedModel, CountModel, SequenceModel, decode_count_model
from trainer import train, train_count, train_seq
import info
import utils
from visualize import visualize


# configuration options
# task: what task to perform (either lex_trans [word-level translation] or 
#        cogs [semantic parsing])
# train: if true, trains a new model from scratch; if false, loads one from disk
# count: what kind of model to train: if true, trains a count-based model
#        (only do this for lex_trans!); if false, trains a neural masked LM
# visualize: if true, runs a visualization step that writes model predictions
#        to an html file

TASK = "lex_trans"
TRAIN = True
COUNT = True
VISUALIZE = True

#TASK = "cogs"
#TRAIN = False
#COUNT = True
#VISUALIZE = True

def main():
    random = np.random.RandomState(0)

    if TASK == "lex_trans":
        from tasks import lex_trans
        data, vocab = lex_trans.load()
        #data = data[:3000]
        model_path = f"tasks/lex_trans/align_model.chk"
        seq_path = f"tasks/lex_trans/seq_model.chk"
        vis_path = f"tasks/lex_trans/vis"
        params = {"lr": 0.00003, "n_batch": 32}
        seq_params = {"lr": 0.003, "n_batch": 32}
    elif TASK == "cogs":
        from tasks import cogs
        data, vocab = cogs.load()
        model_path = f"tasks/cogs/align_model.chk"
        seq_path = f"tasks/cogs/seq_model.chk"
        vis_path = "tasks/cogs/vis"
        params = {"lr": 0.00003, "n_batch": 32}

    if COUNT:
        model = CountModel(vocab)
    else:
        model = Model(vocab).cuda()
    seq_model = SequenceModel(vocab)

    if TRAIN:
        #if COUNT:
        #    train_count(model, vocab, data, model_path)
        #else:
        #    train(model, vocab, data, model_path, random, params)
        train_seq(seq_model, vocab, data, seq_path, random, seq_params)
        assert False
    else:
        if COUNT:
            #with open(model_path, "rb") as reader:
            #    model = pickle.load(reader)
            with bz2.open(model_path, "rt", encoding="utf-8") as reader:
                model = json.load(reader, object_hook=decode_count_model)
                model.vocab = vocab
        else:
            model.load_state_dict(torch.load(model_path))

    if VISUALIZE:
        visualize(model, vocab, data, vis_path)

if __name__ == "__main__":
    main()
