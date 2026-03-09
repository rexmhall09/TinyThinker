import re

UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
THINK_TOKEN = '<think>'
ENDTHINK_TOKEN = '</think>'
CHARS_FILE = 'chars.txt'


class Tokenizer:
    def __init__(self, chars_file=CHARS_FILE, unk_token=UNK_TOKEN, eos_token=EOS_TOKEN):
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.chars_file = chars_file
        self._load_chars()
        self._create_mappings()

    def _load_chars(self):
        with open(self.chars_file, 'r', encoding='utf-8') as f:
            charstext = f.read()
        self.chars = sorted(list(set(charstext)))
        self.chars.extend([self.unk_token, self.eos_token, THINK_TOKEN, ENDTHINK_TOKEN])
        self.vocab_size = len(self.chars)

    def _create_mappings(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi[self.unk_token]
        self.eos_id = self.stoi[self.eos_token]

    def encode(self, s):
        pattern = re.compile(r'(<think>|</think>|<eos>|.)')
        tokens = pattern.findall(s)
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, l):
        return ''.join([self.itos.get(i, self.unk_token) for i in l])
