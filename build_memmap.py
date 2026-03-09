import os

import numpy as np

from tokenizer import Tokenizer

enc_npy = "corpus_int32.npy"
txt = "input.txt"

if os.path.exists(enc_npy):
    print("Encoded file already exists — delete it if you want to rebuild.")
    quit()

tok = Tokenizer()
with open(txt, "r", encoding="utf-8") as f:
    raw = f.read()

arr = np.asarray(tok.encode(raw), dtype=np.int32)
np.save(enc_npy, arr)
print(f"Wrote {arr.size:,} tokens to {enc_npy}  ({arr.nbytes / 1e6:.1f} MB)")
