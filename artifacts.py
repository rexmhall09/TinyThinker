from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

CORPUS_FILENAME = "corpus.npy"
TOKENIZER_FILENAME = "tokenizer.json"
DATASET_META_FILENAME = "dataset_meta.json"
MODEL_CONFIG_FILENAME = "model_config.json"
TRAIN_CONFIG_FILENAME = "train_config.json"
METRICS_FILENAME = "metrics.jsonl"
LAST_CHECKPOINT_FILENAME = "checkpoint_last.pt"
BEST_CHECKPOINT_FILENAME = "checkpoint_best.pt"
FINAL_MODEL_FILENAME = "model_final.pt"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_torch_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    torch.save(payload, Path(path))


def load_torch_artifact(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
