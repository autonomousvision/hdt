import logging
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_from_disk
import datasets
import Config
import torch

def pad_texts(texts):
    max_length = max(len(text) for text in texts)
    return [text + ' '*(max_length-len(text)) for text in texts ]

from transformers import AutoTokenizer

import os
def save_listops_token_ids(tokenizer):
    charset = '[]0123456789SUMMEDMAXMIN[PAD]'
    token_ids = tokenizer(charset)['input_ids']
    import pickle
    with open(os.path.join(Config.TOKENIZER_PATH, 'listops_token_ids.pkl'), 'wb') as writer:
        pickle.dump(token_ids, writer)

class ListOPsDataModule(LightningDataModule):
    def __init__(self):# Use_padding
        super().__init__()
        # self.use_padding = use_padding

        self.tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    def prepare_data(self):
        # Runs on CPU
        # Load -> preprocess -> save
        raw_train_data = load_from_disk(Config.DATA_PATH_TRAIN)

        
        def preprocess_dataset(dataset):
            encoded_inputs = self.tokenizer(dataset["text"], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
            save_listops_token_ids(self.tokenizer)
        
            tokenized_dataset = datasets.Dataset.from_dict({"input_ids": encoded_inputs["input_ids"], 
                                                            # "text": pad_texts(dataset["text"]),
                                                        "attention_mask": encoded_inputs["attention_mask"], 
                                                        "label": dataset["label"]})
            tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'label'])
            # else: 
            #     tokenized_dataset = raw_train_data.map(lambda examples: self.tokenizer(examples['text']), batched=True)
            return tokenized_dataset
        tokenized_dataset = preprocess_dataset(raw_train_data)
        tokenized_dataset.save_to_disk(Config.DATA_PATH_TRAIN_TOKENIZED)

        raw_test_data = load_from_disk(Config.DATA_PATH_TEST, keep_in_memory=Config.KEEP_IN_MEMORY)
        tokenized_dataset = preprocess_dataset(raw_test_data)
        tokenized_dataset.save_to_disk(Config.DATA_PATH_TEST_TOKENIZED)

    def setup(self, stage=None):
        # Called on every device (GPU) seperately
        # Load
        if stage == "fit":
            dataset = load_from_disk(Config.DATA_PATH_TRAIN_TOKENIZED, keep_in_memory=Config.KEEP_IN_MEMORY)
            self._log_tokenization(dataset)
            self.train_data = dataset
            # Switched validation set off
            self.train_data = dataset.select(range(int((1-Config.VALIDATION_SPLIT)*len(dataset))))
            self.validation_data = dataset.select(range(int(1-Config.VALIDATION_SPLIT), len(dataset)))
            return
        if stage == "test":
            self.test_data = load_from_disk(Config.DATA_PATH_TEST_TOKENIZED, keep_in_memory=Config.KEEP_IN_MEMORY)
            self._log_tokenization(self.test_data)
            return
        raise ValueError(f"Stage {stage} not recognized")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=Config.BATCH_SIZE)#, num_workers=Config.N_CONCURRENT_PREPROCESSING_PROCESSES)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=Config.BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=Config.BATCH_SIZE)#, num_workers=Config.N_CONCURRENT_PREPROCESSING_PROCESSES)
    
    def _tokenize_raw_data(self, raw_data):
        return raw_data.map(lambda examples: self.tokenizer(examples['text']), batched=True)

    def _log_tokenization(self, train_dataset):
        # Log overviews so we always know what's going on with weird tokenization tricks
        random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
        input_data = [train_dataset[random_sentence_idx]["input_ids"]]

        logging.info(self.tokenizer.decode(input_data[0]))
        logging.info("... is tokenized into (token boundaries marked by ',') ...")
        logging.info(",".join(self.tokenizer.decode(t) for t in input_data[0]))
        logging.debug(f'Vocab: {self.tokenizer.vocab}')


if __name__=='__main__':
    # print(pad_texts(['1', '22']))
    Config.set_data_paths('ORIGINAL_LISTOPS_025_5_20_1000_5000')
    dm = ListOPsDataModule()
    dm.prepare_data()
    # dm.setup(stage='fit')
    dm.setup(stage='test')
    a = 1