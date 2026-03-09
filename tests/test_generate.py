import torch
from model import GPTLanguageModel


def test_generate_respects_block_size():
    vocab_size = 10
    block_size = 4
    model = GPTLanguageModel(vocab_size, n_embd=8, n_head=2, n_layer=1, block_size=block_size, dropout=0.0)
    idx = torch.zeros((1, block_size + 2), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, block_size + 2 + 3)
