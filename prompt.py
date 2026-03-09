import argparse

import torch
import torch.nn.functional as F

from model import GPTLanguageModel, device, n_embd, n_head, n_layer, dropout, block_size
from tokenizer import Tokenizer

parser = argparse.ArgumentParser(description="Prompt TinyThinker")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=None)
parser.add_argument("--max-tokens", type=int, default=10000)
args = parser.parse_args()

tokenizer = Tokenizer()

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

prompt = ""
while True:
    try:
        prompt += str(input("Prompt: ")) + tokenizer.eos_token
        context_tokens = tokenizer.encode(prompt)
        idx = torch.tensor([context_tokens], dtype=torch.long, device=device)
        generated = idx

        print("Output: ", end='', flush=True)
        for _ in range(args.max_tokens):
            idx_cond = generated[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(args.temperature, 1e-8)
            if args.top_k is not None and args.top_k >= 1:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            next_token_id = idx_next[0].item()
            if next_token_id == tokenizer.eos_id:
                prompt += tokenizer.eos_token
                break
            next_char = tokenizer.itos.get(next_token_id, tokenizer.unk_token)
            prompt += next_char
            print(next_char, end='', flush=True)

        print()
    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except EOFError:
        print("\nExiting.")
        break
