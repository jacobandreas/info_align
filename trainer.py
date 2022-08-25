import torch
from torch import optim
import numpy as np

import info
import utils


N_EPOCH = 1000
N_ITER = 500
#N_EPOCH = 50
#N_ITER = 50


def rand_interval(n, random):
    probs = np.arange(n, 0., -1.)
    probs /= probs.sum()
    start = random.choice(n, p=probs)
    end = random.randint(start, n+1)
    return start, end

def make_example(src, tgt, random, vocab):
    pred_src = random.randint(2)
    #mask_other = random.randint(2)
    other_action = random.randint(3)
    if other_action == 0:
        other_mode = "mask"
    elif other_action == 1:
        other_mode = "ignore"
    elif other_action == 2:
        other_mode = "predict"
    src_mask_start, src_mask_end = rand_interval(len(src), random)
    tgt_mask_start, tgt_mask_end = rand_interval(len(tgt), random)
    #src_mask_start = random.randint(len(src)+1)
    #src_mask_end = random.randint(src_mask_start, len(src)+1)
    #tgt_mask_start = random.randint(len(tgt)+1)
    #tgt_mask_end = random.randint(tgt_mask_start, len(tgt)+1)
    
    if pred_src:
        src_mode = "predict"
        tgt_mode = other_mode
    else:
        tgt_mode = "predict"
        src_mode = other_mode

    return info.mask(
        src,
        tgt,
        src_mask_start,
        src_mask_end,
        tgt_mask_start,
        tgt_mask_end,
        src_mode,
        tgt_mode,
        vocab,
    )


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


def train(model, vocab, data, save_path, random, params):
    random.shuffle(data)
    val_data = data[:500]
    data = data[500:]
    #val_data = data = data[:10]
    #keep_data = []
    #for x, y in data:
    #    xd = vocab.decode(x)
    #    if xd.startswith("run") or xd.startswith("sing") or xd.startswith("dance") or xd.startswith("eat"):
    #        keep_data.append((x, y))
    #val_data = data = keep_data
    #print(len(keep_data))
    model.train()
    opt = optim.AdamW(model.parameters(), lr=params["lr"])
    for i in range(N_ITER):
        print(i)

        total_loss = 0
        for j in range(N_EPOCH):
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
