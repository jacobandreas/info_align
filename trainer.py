import torch
from torch import optim

import info
import utils


LR = 0.0003
N_EPOCH = 200
N_ITER = 500
N_BATCH = 16


def make_example(src, tgt, random, vocab):
    pred_src = random.randint(2)
    mask_other = random.randint(2)
    src_mask_start = random.randint(len(src)+1)
    src_mask_end = random.randint(src_mask_start, len(src)+1)
    tgt_mask_start = random.randint(len(tgt)+1)
    tgt_mask_end = random.randint(tgt_mask_start, len(tgt)+1)
    
    if pred_src:
        src_mode = "predict"
        tgt_mode = "mask" if mask_other else "ignore"
    else:
        tgt_mode = "predict"
        src_mode = "mask" if mask_other else "ignore"

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


def train(model, vocab, data, save_path, random):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    for i in range(N_ITER):
        total_loss = 0
        for j in range(N_EPOCH):
            batch = make_batch(data, random, vocab, n_batch=N_BATCH)
            loss = model(*batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(total_loss / N_EPOCH)

        inp, out = make_batch(data, random, vocab, n_batch=10)
        pred = model.decode(inp)
        for j in range(1):
            print(vocab.decode(inp[j]).strip())
            print(vocab.decode(out[j]).strip())
            print(vocab.decode(pred[j]).strip())
            print()

        torch.save(model.state_dict(), save_path)
