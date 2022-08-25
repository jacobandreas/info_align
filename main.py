#!/usr/bin/env python3

from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

#import morph
#import trans
#import amr

from model import Model, PretrainedModel
from trainer import train
import info
import utils


TASK = "lex_trans"
TRAIN = False
VISUALIZE = True

@torch.no_grad()
def visualize(model, vocab, data, vis_path):
    model.eval()
    #for x in range(1000):
    #    src, tgt = data[x]
    counts = Counter()
    xcounts = Counter()
    ycounts = Counter()
    with open(f"{vis_path}/index.html", "w") as writer:
        print("<html>", file=writer)
        print("<head><meta charset='utf-8'></head>", file=writer)
        print("<body><table>", file=writer)

        for i, (src, tgt) in enumerate(data):
            if src == tgt:
                continue

            src_toks = vocab.decode(src)
            tgt_toks = vocab.decode(tgt)
            print(i, vocab.decode(src), vocab.decode(tgt))
            scores = np.zeros((len(src), len(tgt)))
            choices = []
            src_choices = []
            tgt_choices = []

            for s0 in range(len(src)+1):
                src_choices.append(src_toks[:s0])
                src_choices.append(src_toks[s0:])
                for t0 in range(len(tgt)+1):
                    #this worked:
                    #score_l = info.score_spans(src, tgt, 0, s0, 0, t0, model, vocab)
                    #score_r = info.score_spans(src, tgt, s0, len(src), t0, len(tgt), model, vocab)
                    #outside_l = info.score_spans(src, tgt, 0, s0, t0, len(tgt), model, vocab)
                    #outside_r = info.score_spans(src, tgt, s0, len(src), 0, t0, model, vocab)
                    #outside_l = max(outside_l, 0)
                    #outside_r = max(outside_r, 0)
                    #score = score_l + score_r - outside_l - outside_r

                    score = 1

                    if score >= 1:
                        choices.append((score, (src_toks[:s0], tgt_toks[:t0]), (src_toks[s0:], tgt_toks[t0:])))

                    if s0 == 0:
                        tgt_choices.append(tgt_toks[:t0])
                        tgt_choices.append(tgt_toks[t0:])

            for s in src_choices:
                xcounts[s] += 1
            for t in tgt_choices:
                ycounts[t] += 1

            for choice in sorted(choices):
                score, (s1, t1), (s2, t2) = choice
                if s1 != t1:
                    counts[s1, t1] += score
                    #print(score, s1, t1)
                if s2 != t2:
                    counts[s2, t2] += score
                    #print(score, s2, t2)

            #print()
            #assert False

            #for s0 in range(1, len(src)):
            #    for t0 in range(1, len(tgt)):
            #        #for invert in (True, False):
            #            invert = False
            #            score = info.score_split(src, tgt, 0, s0, len(src), 0, t0, len(tgt), invert, model, vocab)
            #            #scores[:s0, :t0] += score
            #            #scores[s0:, t0:] += score
            #            scores[s0-1, t0-1] = score
            #            choices.append((score, s0, t0))
            #            counts[src_toks[:s0], tgt_toks[:t0]] += score
            #            counts[src_toks[s0:], tgt_toks[t0:]] += score

            #tree = info.parse(src, tgt, model, vocab)
            #for (ss, se), (ts, te), score in tree:
            #    counts[src_toks[ss:se], tgt_toks[ts:te]] += score
            #    print(
            #        "<tr><td>",
            #        src_toks[ss:se],
            #        "</td><td>",
            #        tgt_toks[ts:te],
            #        "</td><td>",
            #        f"{score:.3f}",
            #        "</td></tr>",
            #        file=writer
            #    )
            #    print(src_toks[ss:se], tgt_toks[ts:te], f"{score:.3f}")
            #print("<tr></tr>", file=writer)
            #print()

            #df = pd.DataFrame(scores, index=list(src_toks), columns=list(tgt_toks))
            #plt.clf()
            #sns.heatmap(df)
            #plt.yticks(rotation=0)
            #plt.savefig(f"{vis_path}/{i}.jpg")
            #if len(choices) == 0:
            #    continue
            #score, s0, t0 = max(choices)
            #print(f"<tr><td><img src='{i}.jpg'></td></tr>", file=writer)
            #print("<tr><td>", f"{score:.3f}", src_toks[:s0], "|", src_toks[s0:], "||", tgt_toks[:t0], "|", tgt_toks[t0:], "</td></tr>", file=writer)
            #print("<tr></tr>", file=writer)

            #counts[src_toks[:s0], tgt_toks[:t0]] += score
            #counts[src_toks[s0:], tgt_toks[t0:]] += score

            norm_counts = Counter({(s, t): np.log(counts[s,t]) - np.log(xcounts[s]) - np.log(ycounts[t]) for (s, t), _ in counts.most_common(2000)})

            with open(f"{vis_path}/counts.html", "w") as count_writer:
                print("<html><head><meta charset='utf-8'></head><body><table>", file=count_writer)
                for ((k, v), c) in sorted(counts.most_common(1000), key=lambda x: -x[1]):
                    print("<tr><td>", k, "</td><td>", v, "</td><td>", c, "</tr>", file=count_writer)
                print("</table></body><html>", file=count_writer)
            writer.flush()

            with open(f"{vis_path}/counts_norm.html", "w") as count_writer:
                print("<html><head><meta charset='utf-8'></head><body><table>", file=count_writer)
                for ((k, v), c) in sorted(norm_counts.most_common(1000), key=lambda x: -x[1]):
                    print("<tr><td>", k, "</td><td>", v, "</td><td>", c, "</tr>", file=count_writer)
                print("</table></body><html>", file=count_writer)
            writer.flush()

        #print(vocab.decode(src))
        #print(vocab.decode(tgt))

        #choices = []
        #for sj in (range(1, len(src))):
        #    for tj in range(1, len(tgt)):
        #        for invert in [False, True]:
        #            s = info.score_split(src, tgt, 0, sj, len(src), 0, tj, len(tgt), invert, model, vocab)
        #            choices.append((s, sj, tj, invert))

        #best, sj, tj, invert = min(choices)
        #s1s, s1e = 0, sj
        #s2s, s2e = sj, len(src)
        #t1s, t1e = 0, tj
        #t2s, t2e = tj, len(tgt)
        #if invert:
        #    t1s, t2s = t2s, t1s
        #    t1e, t2e = t2e, t1e
        #print(vocab.decode(src[s1s:s1e]), " | ", vocab.decode(tgt[t1s:t1e]))
        #print(vocab.decode(src[s2s:s2e]), " | ", vocab.decode(tgt[t2s:t2e]))
        #print()

def main():
    random = np.random.RandomState(0)

    #if TASK == "morph":
    #    data, vocab = morph.load()
    #elif TASK == "trans":
    #    data, vocab = trans.load()
    #elif TASK == "amr":
    #    data, vocab = amr.load()
    #model = PretrainedModel(vocab).cuda()

    if TASK == "lex_trans":
        from tasks import lex_trans
        data, vocab = lex_trans.load()
        model_path = f"tasks/lex_trans/align_model.chk"
        vis_path = f"tasks/lex_trans/vis"
        model = Model(vocab).cuda()
        params = {"lr": 0.00003, "n_batch": 32}

    if TRAIN:
        train(model, vocab, data, model_path, random, params)
    else:
        model.load_state_dict(torch.load(model_path))

    if VISUALIZE:
        visualize(model, vocab, data, vis_path)

if __name__ == "__main__":
    main()
