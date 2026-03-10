from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from artifacts import CORPUS_FILENAME, DATASET_META_FILENAME, TOKENIZER_FILENAME, ensure_dir, save_json
from tokenizer import SPECIAL_TOKENS, Tokenizer


def build_dataset(input_path: str | Path, out_dir: str | Path, force: bool = False, min_tokens: int = 2) -> dict[str, object]:
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input text not found: {source_path}")

    text = source_path.read_text(encoding="utf-8")
    tokenizer = Tokenizer.from_text(text)
    encoded = np.asarray(tokenizer.encode(text), dtype=np.int32)
    if encoded.size < min_tokens:
        raise ValueError(f"Dataset produced only {encoded.size} tokens; need at least {min_tokens}")

    output_dir = ensure_dir(out_dir)
    corpus_path = output_dir / CORPUS_FILENAME
    tokenizer_path = output_dir / TOKENIZER_FILENAME
    metadata_path = output_dir / DATASET_META_FILENAME

    outputs = (corpus_path, tokenizer_path, metadata_path)
    if not force and any(path.exists() for path in outputs):
        existing = ", ".join(str(path.name) for path in outputs if path.exists())
        raise FileExistsError(f"Refusing to overwrite existing artifacts in {output_dir}: {existing}. Use --force to replace them.")

    np.save(corpus_path, encoded)
    tokenizer.save(tokenizer_path)

    metadata = {
        "input_path": str(source_path),
        "token_count": int(encoded.size),
        "vocab_size": tokenizer.vocab_size,
        "text_char_count": len(text),
        "unique_char_count": len(set(text)),
        "special_tokens": list(SPECIAL_TOKENS),
        "corpus_file": CORPUS_FILENAME,
        "tokenizer_file": TOKENIZER_FILENAME,
    }
    save_json(metadata_path, metadata)

    return {
        "out_dir": output_dir,
        "corpus_path": corpus_path,
        "tokenizer_path": tokenizer_path,
        "metadata_path": metadata_path,
        "token_count": int(encoded.size),
        "vocab_size": tokenizer.vocab_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TinyThinker dataset artifacts")
    parser.add_argument("--input", default="input.txt", help="Path to the training text file")
    parser.add_argument("--out-dir", default="artifacts/data", help="Directory for generated corpus/tokenizer artifacts")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated artifacts")
    parser.add_argument("--min-tokens", type=int, default=2, help="Minimum number of encoded tokens required")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_dataset(args.input, args.out_dir, force=args.force, min_tokens=args.min_tokens)
    print(
        f"Wrote {result['token_count']:,} tokens and vocab size {result['vocab_size']:,} to {result['out_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
