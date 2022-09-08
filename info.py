import numpy as np
import torch

from model import CountModel


# masks a monolingual (source or tgt) seq
# returns an (inp, out) pair where inp is the input to the model, containing
# a mask token, and out is the contents of the hole (to be predicted by the
# model)
def mask_one(seq, s, p, e, mode, vocab):
    inp = seq[:s]
    out = [vocab.HOLE1]
    if mode == "left":
        inp += [vocab.HOLE1, vocab.SKIP1]
        out += seq[s:p]
    elif mode == "right":
        inp += [vocab.SKIP1, vocab.HOLE1]
        out += seq[p:e]
    else:
        assert mode == "both"
        inp += [vocab.HOLE1]
        out += seq[s:e]
    inp += seq[e:]
    inp += [vocab.END]
    out += [vocab.END]
    return inp, out


# masks a pair of sequences
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
        if tgt_mode == "predict":
            src_out += [vocab.HOLE1]

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


# computes the log-probability of a given masked (inp, out) pair
# (i, ii) and (j, jj) are source and target spans respectively; whether they 
# get masked or predicted is determined by src_mode and tgt_mode (see the `mask`
# function)
def cond(src, tgt, i, ii, j, jj, src_mode, tgt_mode, model, vocab):
    inp, out = mask(src, tgt, i, ii, j, jj, src_mode, tgt_mode, vocab)

    if isinstance(model, CountModel):
        return -model(inp, out)

    inp = torch.tensor(inp)[None, :].cuda()
    out = torch.tensor(out)[None, :].cuda()
    logprob = -model(inp, out)
    return logprob.item()


# computes the log-probability of a masked monolingual sequence
def cond_mono(seq, s, p, e, mode, side, model, vocab):
    inp, out = mask_one(seq, s, p, e, mode, vocab)
    if isinstance(model, CountModel):
        if side == "src":
            return -model.h_src(inp, out)
        else:
            assert side == "tgt", side
            return -model.h_tgt(inp, out)
    inp = torch.tensor(inp)[None, :].cuda()
    out = torch.tensor(out)[None, :].cuda()
    return -model(inp, out)


# computes pointwise mutual information between the source span (s0,
# s1) and the target span (t0, t1) (conditioned on the rest of both sequences)
def score_spans(src, tgt, s0, s1, t0, t1, model, vocab):
    joint = cond(src, tgt, s0, s1, t0, t1, "predict", "predict", model, vocab)
    left = cond(src, tgt, s0, s1, t0, t1, "predict", "mask", model, vocab)
    right = cond(src, tgt, s0, s1, t0, t1, "mask", "predict", model, vocab)
    left = max(joint, left)
    right = max(joint, right)
    score = (joint - left - right) / (-joint + 1e-5)
    return score

# computes conditional pointwise mutual information between the monolingual
# spans (s, p) and (p, e) (conditioned on the rest of the sequence)
def score_mono(seq, s, p, e, side, model, vocab):
    left = cond_mono(seq, s, p, e, "left", side, model, vocab)
    right = cond_mono(seq, s, p, e, "right", side, model, vocab)
    joint = cond_mono(seq, s, p, e, "both", side, model, vocab) - np.log(e - s + 1)
    left = max(joint, left)
    right = max(joint, right)
    assert joint <= left + 1e-5, (joint, left)
    assert joint <= right + 1e-5, (joint, right, vocab.decode(seq), s, p, e)
    #print("sm", joint, left, right)
    score = (joint - left - right) / (-joint + 1e-5)
    return score


# parses a sentence top-down by repeatedly splitting the source into spans (i,
# j), (j, k) and the target into spans (i', j'), (j', k') to maximize 
# pmi(i:j, i':j') + pmi(j:k, j':k') - [ pmi(i:j, j:k) + pmi(i':j', j':k') ]
def parse_greedy(src, tgt, model, vocab):
    out = []
    remaining_spans = [((0, len(src)), (0, len(tgt)), 0)]
    src_toks = vocab.decode(src)
    tgt_toks = vocab.decode(tgt)
    while len(remaining_spans) > 0:
        (ss, se), (ts, te), curr_span_score = remaining_spans.pop(0)
        best_score = 0
        best_split = None
        for sp in range(ss+1, se):
            for tp in range(ts+1, te):
                score_l = score_spans(src, tgt, ss, sp, ts, tp, model, vocab)
                score_r = score_spans(src, tgt, sp, se, tp, te, model, vocab)
                score_xs = score_mono(src, ss, sp, se, "src", model, vocab)
                score_xt = score_mono(tgt, ts, tp, te, "tgt", model, vocab)
                score = (score_l + score_r) - (score_xs + score_xt)
                if score > best_score:
                    best_score = score
                    best_split = (sp, tp)

        out.append(((ss, se), (ts, te), curr_span_score))
        if best_split is not None:
            sp, tp = best_split
            remaining_spans.append(((ss, sp), (ts, tp), best_score))
            remaining_spans.append(((sp, se), (tp, te), best_score))

    return out


# uses dynamic programming to find the pair of aligned constituency trees that
# jointly maximize
# pmi(i:j, i':j') + pmi(j:k, j':k') - [ pmi(i:j, j:k) + pmi(i':j', j':k') ]
# over all span pairs (i, j), (i', j')
def parse(src, tgt, model, vocab):
    scores = {}
    for ss in range(len(src)):
        for se in range(ss, len(src)+1):
            for ts in range(len(tgt)):
                for te in range(ts, len(tgt)+1):
                    score = score_spans(src, tgt, ss, se, ts, te, model, vocab)
                    scores[(ss, se), (ts, te)] = score

    scores_src = {}
    for ss in range(len(src)):
        for se in range(ss, len(src+1)):
            for sp in range(ss, se+1):
                score = score_mono(src, ss, sp, se, "src", model, vocab)
                scores_src[ss, sp, se] = score

    scores_tgt = {}
    for ts in range(len(src)):
        for te in range(ss, len(src+1)):
            for tp in range(ss, se+1):
                score = score_mono(tgt, ts, tp, te, "tgt", model, vocab)
                scores_tgt[ts, tp, te] = score

    tree_scores = {}
    pointers = {}

    for ss in range(len(src)):
        for se in range(ss+1, len(src)+1):
            for t in range(len(tgt)):
                tree_scores[(ss, se), (t, t+1)] = scores[(ss, se), (t, t+1)]
                pointers[(ss, se), (t, t+1)] = None
    for ts in range(len(tgt)):
        for te in range(ts+1, len(tgt)+1):
            for s in range(len(src)):
                tree_scores[(s, s+1), (ts, te)] = scores[(s, s+1), (ts, te)]
                pointers[(s, s+1), (ts, te)] = None

    for sl in range(2, len(src)+1):
        for tl in range(2, len(tgt)+1):
            for ss in range(len(src)-sl+1):
                for ts in range(len(tgt)-tl+1):
                    se = ss + sl
                    te = ts + tl
                    best_score = -np.inf
                    best_split = None
                    for sp in range(ss+1, se):
                        for tp in range(ts+1, te):
                            score_l = tree_scores[(ss, sp), (ts, tp)]
                            score_r = tree_scores[(sp, se), (tp, te)]
                            score = (
                                score_l + score_r
                                + scores[(ss, se), (ts, te)]
                                - (scores_src[ss, sp, se] + scores_tgt[ts, tp, te])
                            )
                            if score > best_score:
                                best_score = score
                                best_split = (sp, tp)
                    tree_scores[(ss, se), (ts, te)] = best_score
                    pointers[(ss, se), (ts, te)] = best_split

    spans = []
    queue = [((0, len(src)), (0, len(tgt)))]
    while len(queue) > 0:
        (ss, se), (ts, te) = queue.pop(0)
        score = scores[(ss, se), (ts, te)]
        pointer = pointers[(ss, se), (ts, te)]
        spans.append(((ss, se), (ts, te), score))
        if pointer is not None:
            sp, tp = pointer
            queue.append(((ss, sp), (ts, tp)))
            queue.append(((sp, se), (tp, te)))
    return spans


# TODO update & use
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
