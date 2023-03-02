import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
import json
import bz2
from model import CountModelEncoder

import info
import utils


#N_EPOCH = 500
#N_ITER = 500
N_EPOCH = 500
N_ITER = 200

# picks an interval uniformly at random from [1, n]
def rand_interval(n, random):
    probs = np.arange(n, 0., -1.)
    probs /= probs.sum()
    start = random.choice(n, p=probs)
    end = random.randint(start, n+1)
    return start, end


# creates a random masked sequence
def make_example(src, tgt, random, vocab):
    if random.randint(2) == 0:
        return make_bi_example(src, tgt, random, vocab)
    else:
        if random.randint(2) == 0:
            return make_mono_example(src, random, vocab)
        else:
            return make_mono_example(tgt, random, vocab)

# creates a random masked bitext sequence
def make_bi_example(src, tgt, random, vocab):
    possibilities = set()
    pred_src = np.random.randint(2)
    other_action = np.random.randint(3)
    if other_action == 0:
        other_mode = "mask"
    elif other_action == 1:
        other_mode = "ignore"
    elif other_action == 2:
        other_mode = "predict"
    if pred_src:
        src_mode = "predict"
        tgt_mode = other_mode
    else:
        tgt_mode = "predict"
        src_mode = other_mode

    for i in range(len(src)+1):
        for ii in range(i, len(src)+1):
            for j in range(len(tgt)+1):
                for jj in range(j, len(tgt)+1):
                        # TODO clean up
                        for x0, x1 in [(i, ii)]:
                            for y0, y1 in [(j, jj)]:
                                if src_mode == "ignore":
                                    x0, x1 = 0, 0
                                if tgt_mode == "ignore":
                                    y0, y1 = 0, 0
                                possibilities.add((src_mode, tgt_mode, x0, x1, y0, y1))
    # We build up the `possibilities` set because some maskings can be generated
    # in multiple ways, but we don't want to double-count them (or joints won't
    # be compatible with conditionals). TODO we should be able to fix this
    # analytically, but I was lazy.
    possibilities = list(possibilities)
    src_mode, tgt_mode, x0, x1, y0, y1 = possibilities[random.randint(len(possibilities))]
    return info.mask(src, tgt, x0, x1, y0, y1, src_mode, tgt_mode, vocab)


# creates a random masked (source- or target-only) sequence.
def make_mono_example(seq, random, vocab):
    s, e = rand_interval(len(seq), random)
    p = random.randint(s, e+1)
    mode = random.choice(["left", "right", "both"])
    return info.mask_one(seq, s, p, e, mode, vocab)


# creates a batch of examples for training on
def make_batch(data, random, vocab, n_batch=32):
    inps = []
    outs = []
    for i in range(n_batch):
        src, tgt = data[random.randint(len(data))]
        if len(src) == 0 or len(tgt) == 0:
            continue
        inp, out = make_example(src, tgt, random, vocab)
        inps.append(inp)
        outs.append(out)
    max_inp_len = max(len(i) for i in inps)
    max_out_len = max(len(o) for o in outs)
    for inp in inps:
        inp.extend([vocab.PAD] * (max_inp_len - len(inp)))
    for out in outs:
        out.extend([vocab.PAD] * (max_out_len - len(out)))
    return torch.tensor(inps).cuda(), torch.tensor(outs).cuda()


# trains a neural sequence model
def train_seq(model, vocab, data, save_path, random, params):
    random.shuffle(data)
    train_data = data[500:]
    val_data = data[:500]
    n_batch = params["n_batch"]
    def collate(batch):
        inp, out = zip(*batch)
        inp = [torch.tensor([vocab.START] + i + [vocab.END]) for i in inp]
        out = [torch.tensor([vocab.START] + o + [vocab.END]) for o in out]
        inp_padded = pad_sequence(inp, padding_value=vocab.PAD)
        out_padded = pad_sequence(out, padding_value=vocab.PAD)
        return inp_padded, out_padded
    train_loader = DataLoader(train_data, batch_size=n_batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=n_batch, collate_fn=collate)
    opt = optim.AdamW(model.parameters(), lr=params["lr"])
    for i in range(N_ITER):
        model.train()
        train_loss = 0
        for inp, out in train_loader:
            loss = model(inp, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inp, out in val_loader:
                loss = model(inp, out)
                val_loss += loss.item()
                sample, = model.sample(inp[:, :1])
                
                example_inp = inp[:, 0].detach().cpu().numpy().tolist()
                print(vocab.decode(example_inp), vocab.decode(sample))
        print(train_loss, val_loss)


# trains a neural model
def train(model, vocab, data, save_path, random, params):
    random.shuffle(data)
    val_data = data[:500]
    data = data[500:]
    model.train()
    opt = optim.AdamW(model.parameters(), lr=params["lr"])
    for i in range(N_ITER):
        print(i)

        total_loss = 0
        for j in tqdm(range(N_EPOCH)):
            batch = make_batch(data, random, vocab, n_batch=params["n_batch"])
            loss = model(*batch).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(total_loss / N_EPOCH)

        with torch.no_grad():
            val_loss = 0
            for j in range(10):
                batch = make_batch(val_data, random, vocab, n_batch=params["n_batch"])
                loss = model(*batch).mean()
                val_loss += loss.item()
            print(val_loss / 10)

            inp, out = make_batch(data, random, vocab, n_batch=10)
            pred = model.decode(inp)
            for j in range(1):
                print(vocab.decode(inp[j]).strip())
                print(vocab.decode(out[j]).strip())
                print(vocab.decode(pred[j]).strip())
                print()

        torch.save(model.state_dict(), save_path)


# trains a count-based model
def train_count(model, vocab, data, save_path):
    for src, tgt in tqdm(data):
        handled = set()
        for i in range(len(src)+1):
            for ii in range(i, len(src)+1):
                for j in range(len(tgt)+1):
                    for jj in range(j, len(tgt)+1):
                        for pred_src in range(2):
                            for other_action in range(3):
                                if other_action == 0:
                                    other_mode = "mask"
                                elif other_action == 1:
                                    other_mode = "ignore"
                                elif other_action == 2:
                                    other_mode = "predict"
                                if pred_src:
                                    src_mode = "predict"
                                    tgt_mode = other_mode
                                else:
                                    tgt_mode = "predict"
                                    src_mode = other_mode

                                # TODO clean up
                                for x0, x1 in [(i, ii)]:
                                    for y0, y1 in [(j, jj)]:
                                        if src_mode == "ignore":
                                            x0, x1 = 0, 0
                                        if tgt_mode == "ignore":
                                            y0, y1 = 0, 0
                                        sig = (x0, x1, y0, y1, src_mode, tgt_mode)
                                        if sig in handled:
                                            continue
                                        handled.add(sig)

                                        x, y = info.mask(src, tgt, x0, x1, y0, y1, src_mode, tgt_mode, vocab)
                                        model.observe(x, y)

        # TODO DOUBLE-CHECK THIS
        # I think we should only count the `both` case once.
        for i_s in range(len(src)+1):
            for i_e in range(i_s, len(src)+1):
                for i_p in range(i_s, i_e+1):
                    for mode in ["left", "right", "both"]:
                        x, y = info.mask_one(src, i_s, i_p, i_e, mode, vocab)
                        model.observe_src(x, y, i_e - i_s + 1)
        for j_s in range(len(tgt)+1):
            for j_e in range(j_s, len(tgt)+1):
                for j_p in range(j_s, j_e+1):
                    for mode in ["left", "right", "both"]:
                        x, y = info.mask_one(tgt, j_s, j_p, j_e, mode, vocab)
                        model.observe_tgt(x, y, j_e - j_s + 1)

    #with open(save_path, "wb") as writer:
    #    pickle.dump(model, writer)
    with bz2.open(save_path, "wb") as writer:
        writer.write(json.dumps(model, cls=CountModelEncoder).encode("utf-8"))
