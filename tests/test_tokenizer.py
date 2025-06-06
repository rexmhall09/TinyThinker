import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from tokenizer import Tokenizer, THINK_TOKEN, ENDTHINK_TOKEN


def test_encode_decode_round_trip():
    tok = Tokenizer()
    text = "hello world"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_think_tokens_preserved():
    tok = Tokenizer()
    text = f"abc{THINK_TOKEN}def{ENDTHINK_TOKEN}ghi"
    tokens = tok.encode(text)
    # Ensure special token ids are present
    assert tok.stoi[THINK_TOKEN] in tokens
    assert tok.stoi[ENDTHINK_TOKEN] in tokens
    assert tok.decode(tokens) == text


def test_unknown_character_decodes_to_unk():
    tok = Tokenizer()
    text = "x\U0001F984y"  # unicorn emoji should be unknown
    tokens = tok.encode(text)
    assert tokens[1] == tok.unk_id
    decoded = tok.decode(tokens)
    assert decoded == f"x{tok.unk_token}y"
