import re
import os
"""Baseline curricula."""
import torch
import datasets
from itertools import chain
import numpy as np
import logging
import contextlib
import transformers
import tempfile
from collections import defaultdict
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
from collections import OrderedDict, abc
"""LMDB dataset to wrap an existing dataset and create a database if necessary."""
import pickle
import platform
import lmdb
import warnings

warnings.filterwarnings("ignore", "The given buffer is not writable", UserWarning)
log = logging.getLogger(__name__)

module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}

def load_tokenizer(tokenizer_path_or_name, seq_length=512, vocab_size=None, cache_dir=None):
    """Load a tokenizer from disk/huggingface. This will never construct a new tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name, trust_remote_code=True)
    except FileNotFoundError:
        tokenizer = download_tokenizer(tokenizer_path_or_name, seq_length, cache_dir)
    if vocab_size is not None and tokenizer.vocab_size != vocab_size:
        raise Warning(f"Loaded tokenizer with vocab_size {tokenizer.vocab_size} incompatible with given vocab.")
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



def sort_tokenized_dataset_by_unigram(tokenized_dataset, tokenizer, num_threads=1, ngram=1, reverse=False):
    # Force unigram counts per token:
    map_setup = dict(
        batched=True,
        batch_size=1024,
        # num_proc=None,  # have to reimplement counting as in-out instead of side effects for this to work. Lets see how slow num_proc=0 is
        load_from_cache_file=False,
        # keep_in_memory=True,
    )

    unigrams_counts_per_token = np.zeros(len(tokenizer), dtype=np.int64)

    def count_unigrams(examples):
        nonlocal unigrams_counts_per_token
        unigrams_counts_per_token += np.bincount(np.asarray(examples["input_ids"]).reshape(-1), minlength=len(tokenizer))

    tokenized_dataset.map(count_unigrams, desc="Counting token unigrams", **map_setup, num_proc=None)

    token_count = sum(unigrams_counts_per_token)
    k = 1
    k_smoothed_probs = (unigrams_counts_per_token + k) / (token_count + k * len(tokenizer))
    log2_probs = np.log2(k_smoothed_probs)

    def return_seq_prob(examples):
        # seq_counts = np.apply_along_axis(np.bincount, axis=1, arr=np.asarray(examples["input_ids"]), minlength=tokenizer.vocab_size)
        # seq_counts = (np.asarray(examples["input_ids"])[:, :,None] == np.arange(0, tokenizer.vocab_size)[None, None, :]).sum(axis=1)  # slower so far
        # logprob_scores = (log2_probs * seq_counts).sum(axis=1) / tokenizer.model_max_length
        # why make hard when can do easy?
        logprob_scores = log2_probs[np.asarray(examples["input_ids"])].sum(axis=1) / tokenizer.model_max_length
        return dict(scores=logprob_scores)

    dataset_probs = tokenized_dataset.map(
        return_seq_prob,
        desc="Computing log probs per sequence",
        remove_columns=tokenized_dataset.column_names,
        **map_setup,
        num_proc=num_threads if num_threads > 0 else None,
    )

    new_order = np.argsort(np.asarray(dataset_probs["scores"]))

    if reverse:
        new_order = new_order[::-1]

    return tokenized_dataset.select(indices=new_order, writer_batch_size=1024)


def sort_tokenized_dataset_by_token(tokenized_dataset, tokenizer, target_token_id, num_threads=1):
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        load_from_cache_file=False,
        # keep_in_memory=True,
    )

    def count_token(examples):
        return dict(counts=(np.asarray(examples["input_ids"]) == target_token_id).sum(axis=1))

    dataset_counts = tokenized_dataset.map(
        count_token,
        desc=f"Counting occurrences of token {tokenizer.decode(target_token_id)}",
        remove_columns=tokenized_dataset.column_names,
        **map_setup,
    )

    new_order = np.argsort(np.asarray(dataset_counts["counts"]))[::-1]

    # Print sentence with most occurrences:
    sentence_idx = int(new_order[0])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("Sentence with most occurrences of token ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    sentence_idx = int(new_order[-1])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("Sentence with least occurrences of token ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    return tokenized_dataset.select(indices=new_order, writer_batch_size=1024)


def sort_tokenized_dataset_by_word_length(tokenized_dataset, tokenizer, num_threads=1):
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        load_from_cache_file=False,
        # keep_in_memory=True,
    )

    def count_word_lengths(examples):
        return dict(lengths=[len(s) for s in tokenizer.batch_decode(torch.as_tensor(examples["input_ids"]))])

    dataset_counts = tokenized_dataset.map(
        count_word_lengths,
        desc="Counting word lengths per sequence",
        remove_columns=tokenized_dataset.column_names,
        **map_setup,
    )

    new_order = np.argsort(np.asarray(dataset_counts["lengths"]))  # shortest sentences first

    # Print sentence with shortest length
    sentence_idx = int(new_order[0])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("Sentence with shortest length ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    sentence_idx = int(new_order[-1])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("and longest ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    return tokenized_dataset.select(indices=new_order, writer_batch_size=1024)


def lookup_dtype(vocab_size):
    if vocab_size < 2**8:
        dtype = torch.uint8
    # would really be neat to have uint16 here between the BERT and GPT encoding sizes
    elif vocab_size < 2**16 // 2:
        dtype = torch.int16
    elif vocab_size < 2**32 // 2:
        dtype = torch.int32
    else:
        dtype = torch.int64
    return dtype


class CachedDataset(torch.utils.data.Dataset):
    """Cache a given dataset into RAM or SDRAM (GPU memory).
    This is only a good idea if you have enough RAM, especially if mapping into SDRAM.
    """

    def __init__(self, dataset, seq_length, vocab_size, num_workers=0, target_device=torch.device("cpu")):
        """Initialize with a given pytorch dataset. The setup dictionary determines cache location and storage type."""
        self.dataset = dataset
        log.info("Caching started ...")
        batch_size = min(len(dataset), 2048)
        cacheloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=transformers.data.data_collator.torch_default_data_collator,
        )
        self.dataset_keys = list(dataset[0].keys())
        seq_lengths = [len(dataset[0][k]) for k in self.dataset_keys]
        assert all([length == seq_lengths[0] for length in seq_lengths])

        # Allocate memory:
        pin = target_device == torch.device("cpu") and torch.cuda.is_available()
        cache_setup = dict(device=target_device, dtype=lookup_dtype(vocab_size), pin_memory=pin)
        self.cache = torch.empty((len(self.dataset), seq_length * 4), **cache_setup)

        pointer = 0
        for data in cacheloader:
            batch_length = data[self.dataset_keys[0]].shape[0]
            data_block = torch.cat([d.to(cache_setup["dtype"]) for d in data.values()], dim=1)
            self.cache[pointer : pointer + batch_length] = data_block
            pointer += batch_length

        self.cache = self.cache.contiguous()
        log.info(f'Dataset successfully cached into {"RAM" if target_device == torch.device("cpu") else "SDRAM"}.')

    def __getitem__(self, index):
        """Get sample, target from cache."""
        sample_data_block = self.cache[index]
        sample_dict = dict(zip(self.dataset_keys, torch.chunk(sample_data_block, len(self.dataset_keys), dim=-1)))
        return sample_dict

    def __len__(self):
        """Length is length of self.dataset."""
        return len(self.dataset)

    def __getattr__(self, name):
        """This is only called if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


def concatenate_entries(dataset, num_entries_in_group, num_threads):
    parellism_flag = os.environ.get("TOKENIZERS_PARALLELISM") or "false"
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def group_texts(examples):
        result = dict()
        for key, entries in examples.items():
            reduced_list = []
            state, num_collected = None, 0
            for entry in entries:
                num_collected += 1
                if num_collected == 1:
                    state = entry
                else:
                    state += entry
                if num_collected == num_entries_in_group:
                    reduced_list.append(state)
                    state, num_collected = None, 0

            result[key] = reduced_list

        return result

    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=True,
    )
    dataset = dataset.map(group_texts, desc="Concatenating examples", **map_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return dataset


def move_stream_to_fixed_map(raw_data_streamed, max_entries_in_raw_dataset, max_raw_chunk_size=1e14):
    """Save streaming dataset to a fixed mapping-style database."""
    # I'm tired of IterableDatasets and will take the performance hit to write them out instead:
    if max_raw_chunk_size > max_entries_in_raw_dataset:
        with tempfile.TemporaryDirectory() as tmpdirname:
            all_text = []
            from tqdm import tqdm
            for v in tqdm(raw_data_streamed, total=max_entries_in_raw_dataset, desc="Pulling stream dataset to memory"):
                all_text.append(v["text"])
            datasets.Dataset.from_dict(dict(text=all_text)).save_to_disk(tmpdirname + "raw_data")
            raw_data_mapped = datasets.load_from_disk(tmpdirname + "raw_data")
        # This used to be only a move into RAM but this breaks memory later using C4:
        # raw_data = datasets.Dataset.from_dict(dict(text=[v["text"] for v in raw_data]))
        return raw_data_mapped
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            mapped_sets = []
            data_in_RAM = defaultdict(list)
            for idx, value_stream in enumerate(raw_data_streamed):
                data_in_RAM["text"].append(value_stream["text"])
                if ((idx + 1) % max_raw_chunk_size == 0) or ((idx - 1) == max_entries_in_raw_dataset):
                    datasets.Dataset.from_dict(data_in_RAM).save_to_disk(tmpdirname + "raw_data" + str(idx))
                    mapped_dataset = datasets.load_from_disk(tmpdirname + "raw_data" + str(idx))
                    log.info(f"Saved temporary copy at idx {idx} of {max_entries_in_raw_dataset} at {tmpdirname + 'raw_data' + str(idx)}.")
                    data_in_RAM["text"] = []
                    mapped_sets.append(mapped_dataset)
        return datasets.concatenate_datasets(mapped_sets)


def create_database(dataset, database_path, cfg_db, target_dtype):
    """Create an LMDB database from the given pytorch dataset.
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    Removed pyarrow dependency
    but that was several iterations of this file over various projects ago...
    """
    if platform.system() == "Linux":
        map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    else:
        raise ValueError("Provide a reasonable default map_size for your operating system and overwrite this part.")
    db = lmdb.open(
        database_path,
        subdir=False,
        map_size=map_size,
        readonly=False,
        meminit=cfg_db.meminit,
        map_async=True,
    )

    txn = db.begin(write=True)
    idx = 0
    if cfg_db.shuffle_while_writing:
        order = torch.randperm(len(dataset)).tolist()  # this might be a problem for larger dataset sizes?
    else:
        order = range(0, len(dataset))
    for indexing in order:
        data = dataset[indexing]
        # serialize super serially, super slow
        data_block = torch.cat([torch.as_tensor(item, dtype=target_dtype) for item in data.values()], dim=0)
        byteflow = data_block.numpy().tobytes()
        txn.put("{}".format(idx).encode("ascii"), byteflow)
        idx += 1

        if idx % cfg_db.write_frequency == 0:
            log.info(f"[{idx} / {len(dataset)}]")
            txn.commit()
            txn = db.begin(write=True)

    # finalize dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]  # How large will these keys be, too large?
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))
    log.info(f"Database written successfully with {len(keys)} entries.")


SPACY_NER_LABELS = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]

def move_to_device(batch, device):
    r"""Puts each data field to the device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch,(list,tuple)):
        return tuple(move_to_device(item,device) for item in batch)
    elif isinstance(batch, abc.Mapping):
        return {key: move_to_device(value,device) for key, value in batch.items()}
    else:
        return batch