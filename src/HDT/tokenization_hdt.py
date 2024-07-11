# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model HDT."""


import os
import re
import warnings
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from itertools import chain
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

class HDTTokenizer:
    """
    A tokenizer wrapper to perform the functions we need upon a basic tokenizer
    """
    def __init__(self, tokenizer, max_document_length):
        self.tokenizer = tokenizer
        self.max_document_length = max_document_length
        self.sec_token_id = self.tokenizer.get_vocab()["<sec>"]
        self.doc_token_id = self.tokenizer.get_vocab()["<doc>"]
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, text, *args, **kwargs):
        # sentences can be either 3 dimension if structure=True or 2 dimension
        assert isinstance(text[0], list)
        return self.tokenize_sparse_document(text, **kwargs)

    def from_pretrained(cls, path):
        return cls(AutoTokenizer.from_pretrained(path))

    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)

    def tokenize_sparse_document(self, document, query=None, global_query=False):
        # variable query is for prompt-like sentences which are not a part of the input document,
        # e.g., question of a document QA task
        # We set the query as the first section of the document and 'global_query' controls whether the query keeps
        # global attention on all the tokens
        # TODO: implement sparse tokenization without structure information
        max_length = self.max_document_length
        model_max_length = self.tokenizer.model_max_length
        self.tokenizer.model_max_length = max_length
        ## additionall return sentence ids, [cls] [sec] mask + section ids, [sec] [doc] mask
        # document = document[:self.max_doc_length]
        keep_pad = 0
        hash_pad = 1e9
        if query is not None:
            if global_query:
                query_ids = self.tokenizer(query, add_special_tokens=False)["input_ids"]
                max_length -= len(query_ids)
            else:
                document = [[query]] + document
        tmp_num_sent = [len(sec) for sec in document]
        num_sent = [sum(tmp_num_sent[:i]) for i in range(1, len(document) + 1)]
        tokenized_inputs = self.tokenizer(list(chain.from_iterable(document)), return_token_type_ids=False, return_attention_mask=False, return_special_tokens_mask=True, truncation=True, add_special_tokens=False)
        tokenized_inputs["input_ids"] = [[self.tokenizer.cls_token_id] + i for i in tokenized_inputs["input_ids"]]
        tokenized_inputs["special_tokens_mask"] = [[1] + i for i in tokenized_inputs["special_tokens_mask"]]
        tokenized_sentences = tokenized_inputs["input_ids"]
        keep_ids = [[0, 0],[0, 1],[1, 1]]
        hash_ids = [[0, 0],[0, 0],[0, 0]]
        position_ids = [[0, 0], [0, 0], [0, 1]]
        input_ids = [self.doc_token_id, self.sec_token_id]
        special_tokens_mask = [1, 1]
        sec_id = 0
        sent_id = 1
        for sent_index, tokenized_sent in enumerate(tokenized_sentences):
            if len(input_ids) >= max_length:
                input_ids = input_ids[:max_length]
                keep_ids = [i[:max_length] for i in keep_ids]
                hash_ids = [i[:max_length] for i in hash_ids]
                position_ids = [i[:max_length] for i in position_ids]
                break
            if True in [sent_index==i for i in num_sent]:
                input_ids += [self.sec_token_id]
                keep_ids[0] += [0]
                keep_ids[1] += [1]
                keep_ids[2] += [1]
                hash_ids[0] += [sent_index]
                sec_id += 1
                hash_ids[1] += [sec_id]
                hash_ids[2] += [0]
                position_ids[0] += [0]
                position_ids[1] += [0]
                position_ids[2] += [sec_id + 1]
                special_tokens_mask += [1]
                sent_id = 1
            input_ids += tokenized_sent
            keep_ids[0] += [1] * len(tokenized_sent)
            keep_ids[1] += ([1] + [0] * (len(tokenized_sent)-1))
            keep_ids[2] += [0] * len(tokenized_sent)
            hash_ids[0] += [sent_index] * len(tokenized_sent)
            hash_ids[1] += [sec_id] * len(tokenized_sent)
            hash_ids[2] += [0] * len(tokenized_sent)
            position_ids[0] += list(range(0, len(tokenized_sent)))
            position_ids[1] += [sent_id] * len(tokenized_sent)
            position_ids[2] += [sec_id+1] * len(tokenized_sent)
            special_tokens_mask += tokenized_inputs["special_tokens_mask"][sent_index]
            sent_id += 1
        if len(input_ids) < max_length:
            real_length = len(input_ids)
            input_ids += (max_length - real_length) * [self.pad_token_id]
            special_tokens_mask += [1] * (max_length - real_length)
            keep_ids = [i + (max_length - real_length) * [keep_pad] for i in keep_ids]
            hash_ids = [i + (max_length - real_length) * [hash_pad] for i in hash_ids]
            position_ids = [i + (max_length - real_length) * [0] for i in position_ids]
        else:
            input_ids = input_ids[:max_length]
            special_tokens_mask = special_tokens_mask[:max_length]
            keep_ids = [i[:max_length] for i in keep_ids]
            hash_ids = [i[:max_length] for i in hash_ids]
            position_ids = [i[:max_length] for i in position_ids]
        assert len(input_ids) == len(special_tokens_mask) == len(keep_ids[0]) == len(hash_ids[0]) == len(keep_ids[1]) == len(hash_ids[1]) == len(keep_ids[2]) == len(hash_ids[2]) == max_length
        if query and global_query:
            input_ids = query_ids + input_ids
            query_keep_ids = [[0] * len(query_ids), [0] * len(query_ids), [1] * len(query_ids)]
            special_tokens_mask = [0] * len(query_ids) + special_tokens_mask
            keep_ids = [query_keep_ids[i] + d for i, d in enumerate(keep_ids)]
            hash_ids = [[0] * len(query_ids) + i for i in hash_ids]
            position_ids = [list(range(len(query_ids))) + i for i in position_ids]
        self.tokenizer.model_max_length = model_max_length
        keep_ids = [np.asarray(i, dtype=np.int32) for i in keep_ids]
        hash_ids = [np.asarray(i, dtype=np.int32) for i in hash_ids]
        position_ids = [np.asarray(i, dtype=np.int32) for i in position_ids]
        # mask = generate_mask_fast(keep_ids, hash_ids)
        # if global_query:
        #     mask[:len(query_ids), :] = 1
        #     mask[:, :len(query_ids)] = 1
        return {"input_ids": np.asarray(input_ids, dtype=np.int32), "special_tokens_mask": np.asarray(special_tokens_mask, dtype=np.int32),
                "keep_ids_0": keep_ids[0], "keep_ids_1": keep_ids[1], "keep_ids_2": keep_ids[2],
                "hash_ids_0": hash_ids[0], "hash_ids_1": hash_ids[1], "hash_ids_2": hash_ids[2],
                "position_ids_0": position_ids[0], "position_ids_1": position_ids[1], "position_ids_2": position_ids[2]}
