from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"
THINK_TOKEN = "<think>"
ENDTHINK_TOKEN = "</think>"
SPECIAL_TOKENS = (UNK_TOKEN, EOS_TOKEN, THINK_TOKEN, ENDTHINK_TOKEN)
TOKEN_PATTERN = re.compile(r"(<think>|</think>|<eos>|.)", re.DOTALL)


@dataclass
class Tokenizer:
    tokens: list[str]
    unk_token: str = UNK_TOKEN
    eos_token: str = EOS_TOKEN

    def __post_init__(self) -> None:
        deduped_tokens = list(dict.fromkeys(self.tokens))
        required_tokens = [self.unk_token, self.eos_token, THINK_TOKEN, ENDTHINK_TOKEN]
        for token in required_tokens:
            if token not in deduped_tokens:
                deduped_tokens.append(token)
        self.tokens = deduped_tokens
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}
        self.unk_id = self.stoi[self.unk_token]
        self.eos_id = self.stoi[self.eos_token]
        self.vocab_size = len(self.tokens)

    @classmethod
    def from_text(cls, text: str, unk_token: str = UNK_TOKEN, eos_token: str = EOS_TOKEN) -> "Tokenizer":
        base_tokens = sorted(set(text))
        return cls(tokens=base_tokens, unk_token=unk_token, eos_token=eos_token)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Tokenizer":
        return cls(tokens=list(payload["tokens"]), unk_token=payload.get("unk_token", UNK_TOKEN), eos_token=payload.get("eos_token", EOS_TOKEN))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "unk_token": self.unk_token,
            "eos_token": self.eos_token,
            "think_token": THINK_TOKEN,
            "endthink_token": ENDTHINK_TOKEN,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        pieces = TOKEN_PATTERN.findall(text)
        return [self.stoi.get(piece, self.unk_id) for piece in pieces]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos.get(token_id, self.unk_token) for token_id in token_ids)
