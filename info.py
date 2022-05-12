import torch

def mask(
        src,
        tgt,
        src_start,
        src_end,
        tgt_start,
        tgt_end,
        src_mode,
        tgt_mode,
        vocab,
):
    assert not (src_mode == "predict" and tgt_mode == "predict")
    assert vocab.HOLE1 == vocab.START

    if src_mode == "ignore":
        src_inp = src
        src_out = []
    elif src_mode == "mask":
        src_inp = src[:src_start] + [vocab.SKIP1] + src[src_end:]
        src_out = []
    else:
        assert src_mode == "predict"
        src_inp = src[:src_start] + [vocab.HOLE1] + src[src_end:]
        src_out = src[src_start:src_end]

    if tgt_mode == "ignore":
        tgt_inp = tgt
        tgt_out = []
    elif tgt_mode == "mask":
        tgt_inp = tgt[:tgt_start] + [vocab.SKIP1] + tgt[tgt_end:]
        tgt_out = []
    else:
        assert tgt_mode == "predict"
        tgt_inp = tgt[:tgt_start] + [vocab.HOLE1] + tgt[tgt_end:]
        tgt_out = tgt[tgt_start:tgt_end]

    inp = src_inp + [vocab.SEP] + tgt_inp + [vocab.END]
    out = [vocab.START] + src_out + tgt_out + [vocab.END]

    return inp, out


def cond(src, tgt, i, ii, j, jj, src_mode, tgt_mode, model, vocab):
    inp, out = mask(src, tgt, i, ii, j, jj, src_mode, tgt_mode, vocab)
    inp = torch.tensor(inp)[None, :].cuda()
    out = torch.tensor(out)[None, :].cuda()
    logprob = model(inp, out)
    return logprob.item()


def score_split(src, tgt, si, sj, sk, ti, tj, tk, invert, model, vocab):
    s1s, s1e = si, sj
    s2s, s2e = sj, sk
    t1s, t1e = ti, tj
    t2s, t2e = tj, tk
    if invert:
        t1s, t2s = t2s, t1s
        t1e, t2e = t2e, t1e

    pmi_1_left = (
        cond(src, tgt, s1s, s1e, t1s, t1e, "predict", "ignore", model, vocab)
        - cond(src, tgt, s1s, s1e, t1s, t1e, "predict", "mask", model, vocab)
    )
    pmi_1_right = (
        cond(src, tgt, s1s, s1e, t1s, t1e, "ignore", "predict", model, vocab)
        - cond(src, tgt, s1s, s1e, t1s, t1e, "mask", "predict", model, vocab)
    )

    pmi_2_left = (
        cond(src, tgt, s2s, s2e, t2s, t2e, "predict", "ignore", model, vocab)
        - cond(src, tgt, s2s, s2e, t2s, t2e, "predict", "mask", model, vocab)
    )
    pmi_2_right = (
        cond(src, tgt, s2s, s2e, t2s, t2e, "ignore", "predict", model, vocab)
        - cond(src, tgt, s2s, s2e, t2s, t2e, "mask", "predict", model, vocab)
    )

    result = (pmi_1_left + pmi_1_right + pmi_2_left + pmi_2_right) / 4
    return result
