from __future__ import annotations

import argparse

import torch

from artifacts import load_torch_artifact
from config import ModelConfig
from model import GPTLanguageModel
from runtime import describe_device, resolve_device
from tokenizer import Tokenizer


def load_model_for_inference(artifact_path: str, device: str) -> tuple[GPTLanguageModel, Tokenizer]:
    artifact = load_torch_artifact(artifact_path, map_location=device)
    tokenizer = Tokenizer.from_dict(artifact["tokenizer"])
    model_config = ModelConfig.from_dict(artifact["model_config"])
    model = GPTLanguageModel(vocab_size=tokenizer.vocab_size, config=model_config).to(device)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model, tokenizer


def generate_completion(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    prompt_with_eos = prompt if prompt.endswith(tokenizer.eos_token) else prompt + tokenizer.eos_token
    context_tokens = tokenizer.encode(prompt_with_eos)
    idx = torch.tensor([context_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id,
    )
    new_token_ids = generated[0, len(context_tokens) :].tolist()
    if tokenizer.eos_id in new_token_ids:
        new_token_ids = new_token_ids[: new_token_ids.index(tokenizer.eos_id)]
    return tokenizer.decode(new_token_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt TinyThinker")
    parser.add_argument("--artifact", default="runs/default/model_final.pt", help="Path to a final model artifact or checkpoint")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default=None, help="Generate once and exit instead of starting interactive mode")
    parser.add_argument("--num-samples", type=int, default=1, help="How many non-interactive samples to print")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, tokenizer = load_model_for_inference(args.artifact, device)
    print(f"Using device: {describe_device(device)}")

    if args.prompt is not None:
        for sample_index in range(args.num_samples):
            completion = generate_completion(model, tokenizer, args.prompt, device, args.max_tokens, args.temperature, args.top_k)
            print(f"Sample {sample_index + 1}: {completion}")
        return 0

    history = ""
    while True:
        try:
            user_prompt = input("Prompt: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            print("\nExiting.")
            break

        history += user_prompt + tokenizer.eos_token
        completion = generate_completion(model, tokenizer, history, device, args.max_tokens, args.temperature, args.top_k)
        print(f"Output: {completion}")
        history += completion
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
