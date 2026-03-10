from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(**payload)


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "artifacts/data"
    run_dir: str = "runs/default"
    batch_size: int = 32
    max_iters: int = 2_000
    lr: float = 3e-4
    eval_interval: int = 100
    eval_steps: int = 20
    save_interval: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    seed: int = 42
    val_fraction: float = 0.1
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    resume: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.dtype not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("dtype must be one of: auto, fp32, fp16, bf16")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        return cls(**payload)
