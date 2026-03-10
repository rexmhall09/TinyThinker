from pathlib import Path

from artifacts import BEST_CHECKPOINT_FILENAME, FINAL_MODEL_FILENAME, LAST_CHECKPOINT_FILENAME, METRICS_FILENAME
from build_memmap import build_dataset
from config import ModelConfig, TrainConfig
from train import train_model


TRAINING_TEXT = "Hello<think>plan</think>world<eos>" * 32


def test_train_model_writes_run_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    input_path.write_text(TRAINING_TEXT, encoding="utf-8")
    build_dataset(input_path, data_dir)

    model_config = ModelConfig(n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0)
    train_config = TrainConfig(
        data_dir=str(data_dir),
        run_dir=str(run_dir),
        batch_size=2,
        max_iters=4,
        lr=1e-3,
        eval_interval=2,
        eval_steps=1,
        save_interval=2,
        grad_clip=1.0,
        weight_decay=0.01,
        seed=123,
        val_fraction=0.2,
        device="cpu",
        dtype="fp32",
        compile=False,
    )

    output_dir = train_model(model_config, train_config)

    assert output_dir == run_dir
    assert (run_dir / LAST_CHECKPOINT_FILENAME).exists()
    assert (run_dir / BEST_CHECKPOINT_FILENAME).exists()
    assert (run_dir / FINAL_MODEL_FILENAME).exists()
    assert (run_dir / METRICS_FILENAME).exists()
