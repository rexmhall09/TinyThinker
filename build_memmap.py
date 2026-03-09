import numpy as np, os, tqdm, mmap
from tokenizer import Tokenizer

enc_npy = "corpus_int32.npy" # 1-byte header + raw int32
txt = "input.txt"

if os.path.exists(enc_npy):
    print("Encoded file already exists — delete it if you want to rebuild.")
    quit()

tok = Tokenizer()
with open(txt, "r", encoding="utf-8") as f:
    raw = f.read()

# Encode and dump as int32 little-endian
arr = np.asarray(tok.encode(raw), dtype=np.int32)
np.save(enc_npy, arr) # produces .npy header + data
print(f"Wrote {arr.size:,} tokens to {enc_npy}  ({arr.nbytes/1e6:.1f} MB)")
