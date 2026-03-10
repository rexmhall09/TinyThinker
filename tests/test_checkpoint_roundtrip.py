from pathlib import Path

import torch

from artifacts import load_torch_artifact, save_torch_artifact
from config import ModelConfig, TrainConfig
from model import GPTLanguageModel
from tokenizer import Tokenizer
from train import build_checkpoint_payload, maybe_resume


def test_checkpoint_resume_round_trip(tmp_path: Path):
    tokenizer = Tokenizer.from_text("hello world")
    model_config = ModelConfig(n_embd=8, n_head=2, n_layer=1, block_size=4, dropout=0.0)
    train_config = TrainConfig(data_dir="data", run_dir=str(tmp_path), batch_size=2, max_iters=4, eval_interval=2, eval_steps=1, save_interval=2, device="cpu", dtype="fp32")

    model = GPTLanguageModel(tokenizer.vocab_size, config=model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = build_checkpoint_payload(model, optimizer, step=3, best_val_loss=1.25, model_config=model_config, train_config=train_config, tokenizer=tokenizer)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_torch_artifact(checkpoint_path, payload)

    restored_model = GPTLanguageModel(tokenizer.vocab_size, config=model_config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    start_step, best_val_loss = maybe_resume(
        restored_model,
        restored_optimizer,
        tmp_path,
        str(checkpoint_path),
        model_config,
        tokenizer,
        lr=1e-3,
        weight_decay=0.01,
        device="cpu",
    )

    assert start_step == 4
    assert best_val_loss == 1.25

    original_state = load_torch_artifact(checkpoint_path)
    assert restored_model.state_dict().keys() == original_state["model_state_dict"].keys()
    for key, value in restored_model.state_dict().items():
        assert torch.equal(value, original_state["model_state_dict"][key])
