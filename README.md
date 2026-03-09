# TinyThinker

A small GPT-style language model with thinking token support. Built from scratch using PyTorch — an improved version of TinyTalker with an enhanced tokenizer, training loop, and a `<think>` token for chain-of-thought reasoning.

## Features

- Character-level tokenizer with special tokens (`<eos>`, `<think>`, `</think>`)
- Transformer architecture with multi-head self-attention
- Memory-mapped data loading for large corpora
- Checkpoint saving during training
- Interactive prompt for text generation

## Project Structure

```
model.py          — GPT model (transformer blocks, attention, feed-forward)
tokenizer.py      — Character-level tokenizer with special token support
train.py          — Training loop with evaluation and checkpointing
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

Adjust hyperparameters at the top of `train.py`:
- `batch_size` — number of parallel sequences (default: 4)
- `max_iters` — training iterations (default: 30000)
- `learning_rate` — optimizer learning rate (default: 3e-4)

The model saves to `model.pth` on completion, with periodic checkpoints in `checkpoints/`.

If `model.pth` already exists, training resumes from that checkpoint.

### Generate Text

```bash
python prompt.py
```

Requires a trained `model.pth`. Enter prompts interactively and the model generates text token by token.

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

## License

MIT
