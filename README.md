# TinyThinker

TinyThinker is a small GPT-style character model with explicit `<think>` / `</think>` reasoning markers, a generated tokenizer artifact, resumable training runs, and a Google Colab notebook that can train and download a model bundle in one pass.

## What changed
- The tokenizer vocabulary is now generated from the actual training text instead of relying on the checked-in `chars.txt` file.
- Training writes a self-contained run directory with checkpoints, configs, metrics, tokenizer metadata, and a final inference artifact.
- Prompting loads the saved artifact directly instead of reconstructing the model from loose side files.
- The default model is right-sized for the included dataset instead of using the old giant hard-coded architecture.
- A Colab notebook is included at `notebooks/train_tinythinker_colab.ipynb` for one-click GPU training and download.

## Repository layout
- `config.py` — shared model and training configuration dataclasses
- `model.py` — GPT model with SDPA-based causal attention
- `tokenizer.py` — generated tokenizer artifact logic
- `build_memmap.py` — builds dataset artifacts from input text
- `train.py` — resumable training entrypoint
- `prompt.py` — inference entrypoint for saved artifacts
- `artifacts.py` — artifact filenames and JSON/checkpoint helpers
- `runtime.py` — device and AMP helpers
- `notebooks/train_tinythinker_colab.ipynb` — Colab notebook for end-to-end training
- `tests/` — targeted unit and smoke tests

## Requirements
- Python 3.10+
- PyTorch 2.1+
- NumPy
- tqdm

Install locally:

```bash
pip install -e .
pip install -e .[dev]
```

## Quick start (local)

### 1) Build dataset artifacts
Use the checked-in `input.txt` by default:

```bash
python build_memmap.py --input input.txt --out-dir artifacts/data --force
```

This writes:
- `artifacts/data/corpus.npy`
- `artifacts/data/tokenizer.json`
- `artifacts/data/dataset_meta.json`

### 2) Train
Run a right-sized default training job:

```bash
python train.py --data-dir artifacts/data --run-dir runs/default
```

Useful overrides:

```bash
python train.py \
  --data-dir artifacts/data \
  --run-dir runs/experiment-1 \
  --batch-size 32 \
  --max-iters 1500 \
  --eval-interval 100 \
  --save-interval 100 \
  --n-embd 384 \
  --n-head 6 \
  --n-layer 6 \
  --block-size 256 \
  --dropout 0.1
```

Each run directory contains at minimum:
- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `model_final.pt`
- `metrics.jsonl`
- `model_config.json`
- `train_config.json`
- `tokenizer.json`

Resume training from the latest run checkpoint:

```bash
python train.py --data-dir artifacts/data --run-dir runs/default
```

Or resume from an explicit checkpoint:

```bash
python train.py --data-dir artifacts/data --run-dir runs/default --resume runs/default/checkpoint_last.pt
```

### 3) Generate text
Prompt from a saved artifact:

```bash
python prompt.py --artifact runs/default/model_final.pt --prompt "Hi!" --num-samples 3
```

Interactive mode:

```bash
python prompt.py --artifact runs/default/model_final.pt
```

## Google Colab: import and run
Open `notebooks/train_tinythinker_colab.ipynb` in Google Colab, switch the runtime to a GPU, and click **Run all**.

The notebook will:
1. clone this repository into `/content/TinyThinker`,
2. install CUDA-enabled PyTorch and Python dependencies,
3. verify GPU availability,
4. use the repo `input.txt` by default (or let you upload/override),
5. build dataset artifacts,
6. train a model into a resumable run directory,
7. print sample generations, and
8. zip and download the training artifacts.

If you want persistence beyond the download, set `USE_GOOGLE_DRIVE = True` in the notebook before running all cells.

## Tests
Run the targeted suite:

```bash
pytest tests/test_tokenizer.py \
  tests/test_generate.py \
  tests/test_build_memmap.py \
  tests/test_checkpoint_roundtrip.py \
  tests/test_train_smoke.py
```

## Notes
- `chars.txt` is no longer the runtime source of truth for the tokenizer.
- The included corpus is tiny, so the default model is intentionally small and trainable.
- The Colab notebook is designed to orchestrate the Python entrypoints in this repo rather than embedding a second training implementation.

## License
MIT
