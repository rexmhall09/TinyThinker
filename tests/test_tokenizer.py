from pathlib import Path

from tokenizer import ENDTHINK_TOKEN, THINK_TOKEN, Tokenizer, UNK_TOKEN


def test_encode_decode_round_trip():
    tok = Tokenizer.from_text("hello world")
    text = "hello world"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_think_tokens_preserved():
    tok = Tokenizer.from_text("abc def ghi")
    text = f"abc{THINK_TOKEN}def{ENDTHINK_TOKEN}ghi"
    tokens = tok.encode(text)
    assert tok.stoi[THINK_TOKEN] in tokens
    assert tok.stoi[ENDTHINK_TOKEN] in tokens
    assert tok.decode(tokens) == text


def test_unknown_character_decodes_to_unk():
    tok = Tokenizer.from_text("xy")
    text = "x\U0001F984y"
    tokens = tok.encode(text)
    assert tokens[1] == tok.unk_id
    decoded = tok.decode(tokens)
    assert decoded == f"x{UNK_TOKEN}y"


def test_save_load_round_trip(tmp_path: Path):
    tokenizer = Tokenizer.from_text("hello there")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = Tokenizer.load(path)
    assert loaded.to_dict() == tokenizer.to_dict()
