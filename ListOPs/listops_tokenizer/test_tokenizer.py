from transformers import AutoTokenizer
from .. import Config


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)
    print(tokenizer('MINMAXMEDSUM[PAD]0123456789[]'))
    print(f'Vocabulary size: {tokenizer.vocab_size}')