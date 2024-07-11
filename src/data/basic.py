import json
import torch
import os.path
from collections.abc import Mapping
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import AutoTokenizer, DefaultDataCollator
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from src.utils.tokenization import construct_tokenizer
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk, IterableDataset
from omegaconf import OmegaConf
from src.utils import *
import configs as CONFIG


def data_collator_with_str(features: List) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            elif isinstance(v, int):
                batch[k] = torch.tensor([i[k] for i in features])
            elif isinstance(v[0], dict):
                batch[k] = [f[k] for f in features]
            elif isinstance(v, str):
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


class BasicDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_collator = DefaultDataCollator()
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.prepared = False
        self.multi_gpu = CONFIG.trainer_config.devices > 1
        self.data_collator = data_collator_with_str

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler = torch.utils.data.distributed.DistributedSampler(self.data_train,
                                                                  drop_last=True) if self.multi_gpu else None
        return DataLoader(self.data_train, batch_size=CONFIG.cfg_exps.batch_size, shuffle=(sampler is None),
                          num_workers=CONFIG.cfg_data.num_proc, collate_fn=self.data_collator, sampler=sampler)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler = torch.utils.data.distributed.DistributedSampler(self.data_val,
                                                                  drop_last=False,
                                                                  shuffle=False) if self.multi_gpu else None
        return DataLoader(self.data_val, batch_size=CONFIG.cfg_exps.batch_size, collate_fn=self.data_collator,
                          num_workers=CONFIG.cfg_data.num_proc, sampler=sampler)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        sampler = torch.utils.data.distributed.DistributedSampler(self.data_test,
                                                                  drop_last=False,
                                                                  shuffle=False) if self.multi_gpu else None
        return DataLoader(self.data_test, batch_size=CONFIG.cfg_exps.batch_size, collate_fn=self.data_collator,
                          num_workers=CONFIG.cfg_data.num_proc, sampler=sampler)

    def _tokenize_raw_data(self, raw_data, **kwargs):
        return self.tokenizer(raw_data["text"])

    def _concatenate_datasets(self, corpus_list):
        raw_data = concatenate_datasets(corpus_list).shuffle(seed=CONFIG.cfg_exps.seed)
        if not isinstance(raw_data, IterableDataset) and CONFIG.cfg_data.max_entries_in_raw_dataset < len(raw_data):
            raw_data = raw_data.select(range(int(CONFIG.cfg_data.max_entries_in_raw_dataset)))
        # raw_data = raw_data.remove_columns([col for col in raw_data.column_names if col not in ["text", "label"]])
        return raw_data

    def _get_tokenizer(self, raw_data):
        tokenizer = construct_tokenizer(raw_data, CONFIG.cfg_data.tok_name, CONFIG.cfg_data.model_max_length, CONFIG.cfg_data.vocab_size, CONFIG.cache_dir,
                                        CONFIG.cfg_data.preprocess_batch_size,
                                        special_tokens=self.special_tokens if hasattr(self, "special_tokens") else {})
        tokenizer.save_pretrained(CONFIG.save_dir)
        return tokenizer

    def _load_tokenizer(self):
        return load_tokenizer(
            CONFIG.cfg_data.tok_name,
            seq_length=CONFIG.cfg_data.model_max_length,
        )

    def _log_tokenization(self, train_dataset):
        # 4) Log overviews so we always know what's going on with weird tokenization tricks
        random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
        input_data = train_dataset[random_sentence_idx]["text"]
            # .squeeze()  # squeeze because hf has leading dim
        dataset_size = len(train_dataset)

        log.info(
            f"Random sentence with seq_length {CONFIG.cfg_data.model_max_length} from dataset of size {dataset_size:,}: ...")
        log.info(input_data)
        log.info("... is tokenized into ...")
        log.info(self.tokenizer.encode([input_data]))