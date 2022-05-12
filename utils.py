PAD = "_"
START = "[START]"
SEP = "[SEP]"
END = "[END]"
HOLE1 = "[HOLE1]"
HOLE2 = "[HOLE2]"
SKIP1 = "[SKIP1]"
SKIP2 = "[SKIP2]"
UNK = "[UNK]"

INIT_VOCAB = {
    PAD: 0,
    START: 1,
    SEP: 2,
    END: 3,
    HOLE1: 4,
    HOLE2: 5,
    SKIP1: 6,
    SKIP2: 7,
    UNK: 8,
}

def decode(seq, vocab, rev_vocab):
    if vocab[END] in seq:
        seq = seq[:seq.index(vocab[END])+1]
    return " ".join(rev_vocab[s] for s in seq)
