# TinyThinker
A improved version of TinyTalker. We have added improvements to tokenizer and trainer aswell as a thinking token

## How To Use:
To Train/Fine-Tune:
1) Add an input.txt file with any text you want it to learn from. (`<eos>` will be converted into an end of statement token and `<think>` or `<\think>` will be think tokens.)
2) Run build_memmap.py to create `corpus_int32.npy` from your input file. This must be done before starting training.
3) Run train.py to train it. (Inside it you can change batch_size, max_iters, learning_rate to march your needs)
4) NOTE: If you have a model.pth file already, it will be loaded from at the start of training and overwritten at the end of training.

To prompt:
Just run prompt.py.
