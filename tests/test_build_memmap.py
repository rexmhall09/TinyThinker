from pathlib import Path

import numpy as np
import pytest

from artifacts import CORPUS_FILENAME, DATASET_META_FILENAME, TOKENIZER_FILENAME
from build_memmap import build_dataset
from tokenizer import Tokenizer


TRAINING_TEXT = "Hello<think>plan</think>world<eos>" * 8


def test_build_dataset_writes_corpus_and_tokenizer(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    out_dir = tmp_path / "data"
    input_path.write_text(TRAINING_TEXT, encoding="utf-8")

    result = build_dataset(input_path, out_dir)

    corpus_path = out_dir / CORPUS_FILENAME
    tokenizer_path = out_dir / TOKENIZER_FILENAME
    metadata_path = out_dir / DATASET_META_FILENAME

    assert result["token_count"] > 0
    assert corpus_path.exists()
    assert tokenizer_path.exists()
    assert metadata_path.exists()

    corpus = np.load(corpus_path)
    tokenizer = Tokenizer.load(tokenizer_path)
    assert corpus.dtype == np.int32
    assert tokenizer.vocab_size < 32


def test_build_dataset_refuses_overwrite_without_force(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    out_dir = tmp_path / "data"
    input_path.write_text(TRAINING_TEXT, encoding="utf-8")

    build_dataset(input_path, out_dir)

    with pytest.raises(FileExistsError):
        build_dataset(input_path, out_dir)
