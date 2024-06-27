from datasets import load_from_disk
import Config
from src.SyntheticDataCreation.ListOps.create_original_listops import create_dataset
from src.BertForListOps.ListOpsDataModule import ListOPsDataModule

# I use this file to create a tokenizer.json file that I then manually edit to contain all the tokens I want. 
dataset_name = create_dataset(0.25, 5, 20, 10000, 10)
Config.set_data_paths(dataset_name)
raw_dataset = load_from_disk(Config.DATA_PATH_TRAIN, keep_in_memory=Config.KEEP_IN_MEMORY)
datamodule = ListOPsDataModule()
datamodule._train_tokenizer(raw_dataset)
datamodule.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(f'Vocabulary size: {datamodule.tokenizer.vocab_size}')
print(datamodule.tokenizer.vocab)
datamodule.tokenizer.save_pretrained(Config.TOKENIZER_PATH + '/example_tokenizer.json')
