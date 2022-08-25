import numpy as np
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
    #assert not (src_mode == "predict" and tgt_mode == "predict")
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


def cond(src, tgt, i, ii, j, jj, src_mode, tgt_mode, model, vocab):
    inp, out = mask(src, tgt, i, ii, j, jj, src_mode, tgt_mode, vocab)
    #print()
    #print("inp ", vocab.decode(inp))
    #print("gold", vocab.decode(out))
    inp = torch.tensor(inp)[None, :].cuda()
    out = torch.tensor(out)[None, :].cuda()
    logprob = -model(inp, out)
    #print("pred", vocab.decode(model.decode(inp)[0]))
    return logprob.item()


def score_spans(src, tgt, s0, s1, t0, t1, model, vocab):
    #cn = -cond(src, tgt, s0, s1, t0, t1, "predict", "predict", model, vocab)
    ##cn = 1
    #c1 = cond(src, tgt, s0, s1, t0, t1, "predict", "ignore", model, vocab)
    #c2 = cond(src, tgt, s0, s1, t0, t1, "predict", "mask", model, vocab)
    #pmi_left = (c1 - c2) / cn
    #c3 = cond(src, tgt, s0, s1, t0, t1, "ignore", "predict", model, vocab)
    #c4 = cond(src, tgt, s0, s1, t0, t1, "mask", "predict", model, vocab) 
    #pmi_right = (c3 - c4) / cn
    #print(vocab.decode(src[s0:s1]), vocab.decode(tgt[t0:t1]))
    #assert -1.1 <= pmi_left <= 1.1, (c1, c2, cn)
    #assert -1.1 <= pmi_right <= 1.1, (c3, c4, cn)
    #print(c1, c2, pmi_left)
    #print(c3, c4, pmi_right)
    #print((pmi_left + pmi_right) / 2)
    #print()
    #return (pmi_left + pmi_right) / 2
    #print("\n---")
    joint = cond(src, tgt, s0, s1, t0, t1, "predict", "predict", model, vocab)
    left = cond(src, tgt, s0, s1, t0, t1, "predict", "mask", model, vocab)
    right = cond(src, tgt, s0, s1, t0, t1, "mask", "predict", model, vocab)
    #assert joint < left, (joint, left)
    #assert joint < right, (joint, right)
    #joint = min(joint, left, right)
    left = max(joint, left)
    right = max(joint, right)
    score = (joint - left - right) / (-joint)
    assert -1 <= score <= 1, (joint, left, right, score)
    return score

def parse(src, tgt, model, vocab):
    scores = {}
    for ss in range(len(src)):
        for se in range(ss, len(src)+1):
            for ts in range(len(tgt)):
                for te in range(ts, len(tgt)+1):
                    score = score_spans(src, tgt, ss, se, ts, te, model, vocab)
                    scores[(ss, se), (ts, te)] = score

    tree_scores = {}
    pointers = {}
    #for s in range(len(src)):
    #    for t in range(len(tgt)):
    #        tree_scores[(s, s+1), (t, t+1)] = scores[(s, s+1), (t, t+1)]
    #        pointers[(s, s+1), (t, t+1)] = None

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
        #print("sl=", sl)
        for tl in range(2, len(tgt)+1):
            #print("tl=", tl)
            for ss in range(len(src)-sl+1):
                #print("ss=", ss)
                for ts in range(len(tgt)-tl+1):
                    #print("ts=", ts)
                    se = ss + sl
                    te = ts + tl
                    best_score = -np.inf
                    best_split = None
                    for sp in range(ss+1, se):
                        for tp in range(ts+1, te):
                            #print(ss, sp, se, "|", ts, tp, te)
                            score_l = tree_scores[(ss, sp), (ts, tp)]
                            score_r = tree_scores[(sp, se), (tp, te)]
                            score_h = scores[(ss, se), (ts, te)]
                            score = score_l + score_r
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

    #print("s1|t1-s1, t1|s1-t1, s2|t2-s2, t2|s2-t2")
    #print(pmi_1_left, pmi_1_right, pmi_2_left, pmi_2_right)
    result = (pmi_1_left + pmi_1_right + pmi_2_left + pmi_2_right) / 4
    #result = max(pmi_1_left, pmi_1_right, pmi_2_left, pmi_2_right)
    return result
