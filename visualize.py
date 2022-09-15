from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch

@torch.no_grad()
def visualize(model, vocab, data, vis_path):
    model.eval()
    counts = Counter()

    for i, (src, tgt) in enumerate(data):
        if src == tgt:
            continue
        src_toks = tuple(vocab.decode(src).split())
        tgt_toks = tuple(vocab.decode(tgt).split())

        print()
        print(i, vocab.decode(src), vocab.decode(tgt))

        for (s0, s1), (t0, t1), score in info.parse_greedy(src, tgt, model, vocab):
            print(src_toks[s0:s1], tgt_toks[t0:t1], score)
            counts[src_toks[s0:s1], tgt_toks[t0:t1]] += score

        with open(f"{vis_path}/counts.html", "w") as count_writer:
            print("<html><head><meta charset='utf-8'></head><body><table>", file=count_writer)
            for ((k, v), c) in sorted(counts.most_common(1000), key=lambda x: -x[1]):
                print("<tr><td>", k, "</td><td>", v, "</td><td>", c, "</tr>", file=count_writer)
            print("</table></body><html>", file=count_writer)
        writer.flush()
