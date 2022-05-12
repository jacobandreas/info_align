import os

def load(vocab):
    data = []
    with open(f"{os.path.dirname(__file__)}/spa") as reader:
        for line in reader:
            lemma, inflected, tags = line.strip().split("\t")
            for token in list(inflected) + list(lemma) + tags.split(";"):
                if token not in vocab:
                    vocab[token] = len(vocab)
            src = [vocab[token] for token in list(inflected)]
            tgt = [vocab[token] for token in list(lemma) + tags.split(";")]
            data.append((src, tgt))

            if len(data) == 100:
                break

            #if len(data) == 10000:
            #    break
    return data
