import copy
import torch
import logging
import contextlib
from itertools import chain
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex, processors, AddedToken
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def _get_sane_token_args():
    return dict(
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
    )


def _get_sane_normalizers(force_english_keyboard=False, force_lowercase=False, strip_accents=False, whitespace_escape=False):
    """original rules as in XLNET with optional modifications. force_english_keyboard is actually an ascii normalization."""
    normalize_ops = []
    normalize_ops.append(normalizers.Replace("``", '"'))
    normalize_ops.append(normalizers.Replace("''", '"'))
    normalize_ops.append(normalizers.NFD() if strip_accents else normalizers.NFKC())
    if force_lowercase:
        normalize_ops.append(normalizers.Lowercase())
    if strip_accents:
        normalize_ops.append(normalizers.StripAccents())
    normalize_ops.append(normalizers.Replace(Regex(" {2,}"), " "))
    if force_english_keyboard:
        normalize_ops.append(normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""))  # start from 00 instead of 1F to include tab
    if whitespace_escape:
        normalize_ops.append(normalizers.Replace(Regex(" "), "一"))  # ▁ this might kill some of the tokenization schemes...
        # using yi in the in the previous regex because huggingface does not split on yi, but would split on bigunderscore
    return normalizers.Sequence(normalize_ops)


def _construct_tokenizer(raw_datasets, tok_name, model_max_length, vocab_size, batch_size, known_tokens=[], special_tokens={}, max_entries=1e5):
    """The actual generation instructions for a new tokenizer. Might make this more scriptable in the future...
    Follows closely along with https://huggingface.co/course/chapter6"""
    len_dataset = len(raw_datasets)

    def batch_iterator(batch_size):
        for entry in raw_datasets.select(range(int(min(max_entries, len(raw_datasets))))).iter(batch_size):
            if type(entry["text"][0]) == list:
                yield list(chain.from_iterable(chain.from_iterable(entry["text"])))
            else:
                yield entry["text"]

    known_tokens = [AddedToken(token, single_word=True) for token in known_tokens]
    special_token_args = {**_get_sane_token_args(), **dict(zip(special_tokens.keys(), [AddedToken(token, single_word=True, normalized=False) for token in special_tokens.values()]))}
    normalizer_sequence = _get_sane_normalizers()
    # Outline tokenizer rules:
    if tok_name == "Unigram":  # without the sentencepice part
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # tokenizer.decoder = None
        special_tokens = list(set(v for k, v in special_token_args.items()))

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=special_token_args["unk_token"],
        )
    elif tok_name == "BPE":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_tokens(known_tokens)

        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, special_tokens=list(set(special_token_args.values()))
        )
    elif tok_name == "WordPiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token=special_token_args["unk_token"]))
        tokenizer.add_tokens(known_tokens)

        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=list(set(special_token_args.values())))
    elif tok_name == "WordPieceBERT":
        # Sanity check tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizers.BertNormalizer()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=list(set(special_token_args.values())))
    elif tok_name == "WordLevel":
        tokenizer = Tokenizer(models.WordLevel(unk_token=special_token_args["unk_token"]))
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=list(set(special_token_args.values())))
    elif tok_name == "SentencePieceBPE":
        """ref https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py"""
        tokenizer = Tokenizer(models.BPE())
        tokenizer.add_tokens(known_tokens)

        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, special_tokens=list(set(special_token_args.values()))
        )
    elif tok_name == "SentencePieceUnigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.add_tokens(known_tokens)
        tokenizer.normalizer = normalizer_sequence
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
        special_tokens = list(set(v for k, v in special_token_args.items()))

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=special_token_args["unk_token"],
        )
    else:
        raise ValueError(f"Invalid tokenization strategy {tok_name} given.")

    # Construct tokenizer
    tokenizer.train_from_iterator(batch_iterator(batch_size), trainer=trainer, length=int(min(max_entries, len(raw_datasets))))

    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(f"Tokenizer generation failure. Vocab size of trained tokenizer is {tokenizer.get_vocab_size()}.")

    # Postprocess:
    cls_token_id = tokenizer.token_to_id("<cls>")
    sep_token_id = tokenizer.token_to_id("<sep>")

    # Generate template:
    single_template = "$A"
    single_template = "<cls> " + single_template
    tokenizer.post_processor = processors.TemplateProcessing(
        single=single_template,
        pair=f"<cls>:0 $A:0 <sep>:0 $B:1 <sep>:1",
        special_tokens=[("<cls>", cls_token_id), ("<sep>", sep_token_id)],
    )
    # Wrap into fast codebase
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
        **special_token_args,
    )
    return wrapped_tokenizer


def load_tokenizer(tokenizer_path_or_name, seq_length=512, vocab_size=None, cache_dir=None):
    """Load a tokenizer from disk/huggingface. This will never construct a new tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, trust_remote_code=True)
    except FileNotFoundError:
        tokenizer = download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir)
    if vocab_size is not None and len(tokenizer) != vocab_size:
        raise Warning(f"Loaded tokenizer with vocab_size {len(tokenizer)} incompatible with given vocab.")
    return tokenizer


@contextlib.contextmanager
def main_process_first():
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.
    This is a stripped-down version of the the huggingface context manager from commit 2eb7bb15e771f13192968cd4657c78f76b0799fe
    """
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield


def download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir=None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, cache_dir=cache_dir)
        tokenizer.model_max_length = seq_length
    except OSError as error_msg:
        raise OSError(f"Invalid huggingface tokenizer {tokenizer_path_or_name} given: {error_msg}")
    return tokenizer


def construct_tokenizer(raw_datasets, tok_name, model_max_length, vocab_size, cache_dir, batch_size, known_tokens=[], special_tokens={}):
    """Construct a new tokenizer. This may include downloading from huggingface."""
    if tok_name not in ["BPE", "Unigram", "WordLevel", "WordPiece", "WordPieceBERT", "SentencePieceUnigram", "SentencePieceBPE"]:
        tokenizer = load_tokenizer(tok_name, model_max_length, cache_dir=cache_dir)
        # Here we don't add special tokens manually as we are loading trained tokenizers
        try:
            for tok_name, tok in special_tokens.items():
                if tok_name == "additional_special_tokens" and isinstance(tok, list):
                    tokenizer.add_special_tokens({tok_name: tok}, replace_additional_special_tokens=False)
                elif tok_name in tokenizer.special_tokens_map.keys():
                    tokenizer.add_special_tokens({tok_name: tok})
                else:
                    tokenizer.add_tokens([tok])
            tokenizer.__setattr__("model_max_length", model_max_length)
        except AttributeError as e:
            logging.warning(f"Add special tokens is not supprted by this tokenizer, see the error message: \n {e}")
    else:
        tokenizer = _construct_tokenizer(raw_datasets, tok_name, model_max_length, vocab_size, batch_size, known_tokens, special_tokens)
    tokenizer.name = f"{tok_name.rstrip('/').split('/')[-1]}-{vocab_size}.json"
    return tokenizer