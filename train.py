from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import tqdm

from artifacts import (
    BEST_CHECKPOINT_FILENAME,
    CORPUS_FILENAME,
    FINAL_MODEL_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    METRICS_FILENAME,
    MODEL_CONFIG_FILENAME,
    TOKENIZER_FILENAME,
    TRAIN_CONFIG_FILENAME,
    append_jsonl,
    ensure_dir,
    load_torch_artifact,
    save_json,
    save_torch_artifact,
)
from config import ModelConfig, TrainConfig
from model import GPTLanguageModel
from runtime import autocast_context, describe_device, resolve_amp_dtype, resolve_device, seed_all
from tokenizer import Tokenizer


def load_dataset(data_dir: str | Path) -> tuple[Tokenizer, np.memmap]:
    directory = Path(data_dir)
    tokenizer_path = directory / TOKENIZER_FILENAME
    corpus_path = directory / CORPUS_FILENAME
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found: {tokenizer_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus artifact not found: {corpus_path}")
    tokenizer = Tokenizer.load(tokenizer_path)
    corpus = np.load(corpus_path, mmap_mode="r")
    return tokenizer, corpus


def split_corpus(corpus: np.memmap, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    split_index = int(len(corpus) * (1.0 - val_fraction))
    train_data = corpus[:split_index]
    val_data = corpus[split_index:]
    return train_data, val_data


def validate_corpus_sizes(train_data: np.ndarray, val_data: np.ndarray, block_size: int) -> None:
    if len(train_data) <= block_size:
        raise ValueError(f"train_data (size={len(train_data)}) must be larger than block_size ({block_size})")
    if len(val_data) <= block_size:
        raise ValueError(f"val_data (size={len(val_data)}) must be larger than block_size ({block_size})")


def get_batch(split_data: np.ndarray, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    upper_bound = len(split_data) - block_size
    if upper_bound <= 0:
        raise ValueError("split_data is too short for the requested block_size")
    starts = np.random.randint(0, upper_bound, size=batch_size)
    x_np = np.stack([split_data[start : start + block_size] for start in starts]).astype(np.int64, copy=False)
    y_np = np.stack([split_data[start + 1 : start + block_size + 1] for start in starts]).astype(np.int64, copy=False)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: str,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split_name, split_data in (("train", train_data), ("val", val_data)):
        split_losses = []
        for _ in range(train_config.eval_steps):
            xb, yb = get_batch(split_data, model_config.block_size, train_config.batch_size, device)
            with autocast_context(device, amp_dtype):
                _, loss = model(xb, yb)
            split_losses.append(float(loss.item()))
        losses[split_name] = float(sum(split_losses) / len(split_losses))
    model.train()
    return losses


def build_checkpoint_payload(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
    tokenizer: Tokenizer,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
    }


def build_inference_artifact(
    model: GPTLanguageModel,
    step: int,
    best_val_loss: float,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": model_config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
    }


def maybe_resume(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    resume_path: str | None,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple[int, float]:
    checkpoint_path = Path(resume_path) if resume_path else run_dir / LAST_CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return 1, math.inf

    checkpoint = load_torch_artifact(checkpoint_path, map_location=device)
    checkpoint_model_config = ModelConfig.from_dict(checkpoint["model_config"])
    if checkpoint_model_config.to_dict() != model_config.to_dict():
        raise ValueError("Checkpoint model_config does not match requested model configuration")
    if checkpoint["tokenizer"] != tokenizer.to_dict():
        raise ValueError("Checkpoint tokenizer metadata does not match the active dataset tokenizer")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = weight_decay

    start_step = int(checkpoint["step"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
    print(f"Resuming from {checkpoint_path} at step {start_step}")
    return start_step, best_val_loss


def train_model(model_config: ModelConfig, train_config: TrainConfig) -> Path:
    seed_all(train_config.seed)
    device = resolve_device(train_config.device)
    amp_dtype = resolve_amp_dtype(device, train_config.dtype)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tokenizer, corpus = load_dataset(train_config.data_dir)
    train_data, val_data = split_corpus(corpus, train_config.val_fraction)
    validate_corpus_sizes(train_data, val_data, model_config.block_size)

    run_dir = ensure_dir(train_config.run_dir)
    save_json(run_dir / MODEL_CONFIG_FILENAME, model_config.to_dict())
    save_json(run_dir / TRAIN_CONFIG_FILENAME, train_config.to_dict())
    tokenizer.save(run_dir / TOKENIZER_FILENAME)

    model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, config=model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    start_step, best_val_loss = maybe_resume(
        model,
        optimizer,
        run_dir,
        train_config.resume,
        model_config,
        tokenizer,
        train_config.lr,
        train_config.weight_decay,
        device,
    )

    training_model: torch.nn.Module = model
    if train_config.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requested but this torch build does not provide it")
        training_model = torch.compile(model)

    print(f"Using device: {describe_device(device)}")
    print(f"Parameters: {sum(param.numel() for param in model.parameters()) / 1e6:.2f}M")
    if amp_dtype is not None:
        print(f"AMP dtype: {amp_dtype}")

    metrics_path = run_dir / METRICS_FILENAME
    for step in tqdm.tqdm(range(start_step, train_config.max_iters + 1), initial=start_step - 1, total=train_config.max_iters):
        xb, yb = get_batch(train_data, model_config.block_size, train_config.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            _, loss = training_model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        if step % train_config.eval_interval == 0 or step == train_config.max_iters:
            losses = estimate_loss(training_model, train_data, val_data, model_config, train_config, device, amp_dtype)
            record = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
            }
            append_jsonl(metrics_path, record)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_torch_artifact(
                    run_dir / BEST_CHECKPOINT_FILENAME,
                    build_checkpoint_payload(model, optimizer, step, best_val_loss, model_config, train_config, tokenizer),
                )

        if step % train_config.save_interval == 0 or step == train_config.max_iters:
            save_torch_artifact(
                run_dir / LAST_CHECKPOINT_FILENAME,
                build_checkpoint_payload(model, optimizer, step, best_val_loss, model_config, train_config, tokenizer),
            )

    save_torch_artifact(
        run_dir / FINAL_MODEL_FILENAME,
        build_inference_artifact(model, train_config.max_iters, best_val_loss, model_config, tokenizer),
    )
    print(f"Saved run artifacts to {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    model_defaults = ModelConfig()
    train_defaults = TrainConfig()

    parser = argparse.ArgumentParser(description="Train TinyThinker")
    parser.add_argument("--data-dir", default=train_defaults.data_dir, help="Directory containing corpus.npy and tokenizer.json")
    parser.add_argument("--run-dir", default=train_defaults.run_dir, help="Directory to store checkpoints and final artifacts")
    parser.add_argument("--resume", default=train_defaults.resume, help="Optional checkpoint path to resume from")
    parser.add_argument("--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument("--max-iters", type=int, default=train_defaults.max_iters)
    parser.add_argument("--lr", type=float, default=train_defaults.lr)
    parser.add_argument("--eval-interval", type=int, default=train_defaults.eval_interval)
    parser.add_argument("--eval-steps", type=int, default=train_defaults.eval_steps)
    parser.add_argument("--save-interval", type=int, default=train_defaults.save_interval)
    parser.add_argument("--grad-clip", type=float, default=train_defaults.grad_clip)
    parser.add_argument("--weight-decay", type=float, default=train_defaults.weight_decay)
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument("--val-fraction", type=float, default=train_defaults.val_fraction)
    parser.add_argument("--device", default=train_defaults.device)
    parser.add_argument("--dtype", default=train_defaults.dtype)
    parser.add_argument("--compile", action="store_true", default=train_defaults.compile)

    parser.add_argument("--n-embd", type=int, default=model_defaults.n_embd)
    parser.add_argument("--n-head", type=int, default=model_defaults.n_head)
    parser.add_argument("--n-layer", type=int, default=model_defaults.n_layer)
    parser.add_argument("--block-size", type=int, default=model_defaults.block_size)
    parser.add_argument("--dropout", type=float, default=model_defaults.dropout)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = ModelConfig(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        lr=args.lr,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        save_interval=args.save_interval,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_fraction=args.val_fraction,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        resume=args.resume,
    )
    train_model(model_config, train_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
