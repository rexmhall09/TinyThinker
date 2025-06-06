# TinyThinker
A improved version of TinyTalker. We have added improvements to tokenizer and trainer aswell as a thinking token

## How To Use:
To Train/Fine-Tune:
1) Add an input.txt file with any text you want it to learn from. (`<eos>` will be converted into an end of statement token and `<think>` or `<\think>` will be think tokens.)
2) Run train.py to train it. (Inside it you can change batch_size, max_iters, learning_rate to march your needs)
3) NOTE: If you have a model.pth file already, it will be loaded from at the start of training and overwritten at the end of training.

To prompt:
Just run prompt.py.

## Running Tests
Install pytest with `pip install pytest` if you don't already have it. Then execute:

```bash
pytest
```

from the repository root to run the unit tests.
