# TinyThinker

A small GPT-style language model with thinking token support. Built from scratch using PyTorch — an improved version of TinyTalker with an enhanced tokenizer, training loop, and a `<think>` token for chain-of-thought reasoning.

## Features

- Character-level tokenizer with special tokens (`<eos>`, `<think>`, `</think>`)
- Transformer architecture with batched multi-head self-attention and GELU activation
- Memory-mapped data loading for large corpora
- Checkpoint saving during training with loss logging
- Gradient clipping for stable training
- Interactive prompt with temperature and top-k sampling controls
- CLI arguments for all hyperparameters

## Project Structure

```
model.py          — GPT model (transformer blocks, batched attention, feed-forward)
tokenizer.py      — Character-level tokenizer with special token support
train.py          — Training loop with evaluation, checkpointing, and loss logging
build_memmap.py   — Preprocesses input.txt into a memory-mapped numpy array
prompt.py         — Interactive CLI for prompting a trained model
chars.txt         — Character vocabulary
tests/            — Unit tests
```

## Requirements

- Python 3.12+
- PyTorch
- NumPy
- tqdm

Install dependencies:

```bash
pip install torch numpy tqdm
```

## Usage

### Prepare Training Data

1. Add your training text to `input.txt`. Use `<eos>` for end-of-statement markers and `<think>`/`</think>` for thinking tokens.
2. Preprocess the corpus:

```bash
python build_memmap.py
```

This creates `corpus_int32.npy`, a memory-mapped numpy array used during training.

### Train

```bash
python train.py
```

All hyperparameters are configurable via CLI:

```bash
python train.py --batch-size 8 --max-iters 50000 --lr 1e-4 --grad-clip 1.0
```

| Flag              | Default   | Description                        |
|-------------------|-----------|------------------------------------|
| `--batch-size`    | 4         | Parallel sequences per step        |
| `--max-iters`     | 30000     | Total training iterations          |
| `--lr`            | 3e-4      | Learning rate                      |
| `--eval-interval` | 100       | Steps between loss evaluations     |
| `--save-interval` | 1000      | Steps between checkpoint saves     |
| `--grad-clip`     | 1.0       | Max gradient norm                  |
| `--seed`          | 42        | Random seed                        |
| `--corpus`        | corpus_int32.npy | Path to preprocessed data   |

The model saves to `model.pth` on completion, with periodic checkpoints in `checkpoints/`. If `model.pth` already exists, training resumes from that checkpoint.

### Generate Text

```bash
python prompt.py
```

Control generation quality with sampling parameters:

```bash
python prompt.py --temperature 0.7 --top-k 50
```

| Flag            | Default | Description                              |
|-----------------|---------|------------------------------------------|
| `--temperature` | 0.8     | Lower = more focused, higher = more random |
| `--top-k`       | None    | Limit sampling to top K tokens           |
| `--max-tokens`  | 10000   | Maximum tokens to generate per response  |

### Run Tests

```bash
pip install pytest
pytest
```

## Model Architecture

| Parameter    | Value |
|-------------|-------|
| Embedding   | 1600  |
| Heads       | 25    |
| Layers      | 48    |
| Block size  | 1028  |
| Dropout     | 0.2   |
| Activation  | GELU  |

## License

MIT
