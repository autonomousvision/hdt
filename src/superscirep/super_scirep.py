import os
import torch
import json
import numpy as np
from typing import Dict, Union, List
from itertools import chain
from collections import OrderedDict,abc
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import logging
from .evaluation.evaluator import IREvaluator, SupervisedEvaluator, SupervisedTask
from .evaluation.few_shot_evaluator import FewShotEvaluator
from .reviewer_matching import ReviewerMatchingEvaluator
from .evaluation.eval_datasets import SimpleDataset, IRDataset
from transformers import AutoTokenizer
from src.lightning_modules.basics import BudgetModel
logger = logging.getLogger(__name__)

@torch.no_grad()
class SciRep4Model:
    def __init__(self, model, tokenizer_path, variant: str = "default",
                 adapters_load_from: Union[str, Dict] = None, fusion_load_from: str = None,
                 use_ctrl_codes: bool = False, task_id: Union[str, Dict] = None,
                 all_tasks: list = None, hidden_dim: int = 768, max_len: int = 512, use_fp16=False, **kwargs):
        self.variant = variant
        self.encoder = model
        self.encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.use_ctrl_codes = use_ctrl_codes
        self.reqd_token_idx = 0 if not use_ctrl_codes else 1
        self._task_id = task_id
        if self._task_id:
            if use_ctrl_codes:
                logger.info(f"Control code used: {self._task_id}")
            elif variant != "default":
                logger.info(f"Task id used: {self._task_id}")

        self.hidden_dim = hidden_dim
        self.max_length = max_len
        self.use_fp16 = use_fp16

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        if self.use_ctrl_codes:
            logger.info(f"Control code used: {value}")
        elif self.variant != "default":
            logger.info(f"Task id used: {value}")
        self._task_id = value

    def __call__(self, batch, batch_ids=None):
        def append_ctrl_code(batch, batch_ids):
            if type(self._task_id) == dict:
                batch = [f"{self.task_id['query']} {text}" if bid[1] == "q" else f"{self.task_id['candidates']} {text}"
                         for text, bid in zip(batch, batch_ids)]
            else:
                batch = [f"{self.task_id} {text}" for text in batch]
            return batch

        batch = [batch] if type(batch) == str else batch
        batch_ids = [] if not batch_ids else batch_ids
        if self.use_ctrl_codes:
            batch = append_ctrl_code(batch, batch_ids)
        inputs = self.tokenizer(batch, padding="max_length", truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=self.max_length)
        if type(inputs["input_ids"]) == list:
            inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        inputs = move_to_device(inputs, self.encoder.device)
        if self.variant == "default":
            output = self.encoder(**inputs, output_hidden_states=True)
        elif type(self._task_id) != dict:
            output = self.encoder(task_id=self._task_id, **inputs)
        else:
            x = inputs["input_ids"]
            output = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to("cuda")
            q_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "q"])
            c_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "c"])

            if not q_idx.shape[0]:
                output = self.encoder(task_id=self._task_id["candidates"], **inputs)
            elif not c_idx.shape[0]:
                output = self.encoder(task_id=self._task_id["query"], **inputs)
            else:
                for i, v in enumerate(sorted(self._task_id.values())):
                    curr_input_idx = q_idx if v == "[QRY]" else c_idx
                    curr_input = x[curr_input_idx]
                    curr_output = self.encoder(task_id=v, input_ids=curr_input,
                                               attention_mask=inputs["attention_mask"][curr_input_idx])
                    try:
                        output[curr_input_idx] = curr_output  # adapters
                    except:
                        output[curr_input_idx] = curr_output.last_hidden_state  # pals
        try:
            if hasattr(output, "pooler_output"):
                embedding = output.pooler_output
            elif hasattr(output, "representations"):
                embedding = output.representations
            else:
                embedding = output.hidden_states[-1][:, self.reqd_token_idx, :]  # cls token
        except:
            embedding = output[:, self.reqd_token_idx, :]  # cls token
        return embedding.half() if self.use_fp16 else embedding


class SciRep4HModel:
    def __init__(self, model, tokenizer_path, variant: str = "default", sparse=False, document=True,
                 use_ctrl_codes: bool = False, task_id: Union[str, Dict] = None, max_len=8192, use_fp16=False):
        self.variant = variant
        # self.encoder = EncoderFactory(base_checkpoint, adapters_load_from, fusion_load_from, all_tasks).get_encoder(
        #     variant)
        self.config = model.base_model.config
        self.encoder = model
        self.document = document
        self.encoder.eval()
        # tokenizer_checkpoint = f"{base_checkpoint}/tokenizer" if os.path.isdir(base_checkpoint) else base_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.sparse = sparse
        # from src.datamodules.tokenizers import HSTokenizer
        # self.htrans_tokenizer = HSTokenizer(self.tokenizer, getattr(self.config, "max_sent_length", None), getattr(self.config, "max_sec_length", None), getattr(self.config, "max_doc_length", None), structure, sparse=self.sparse, max_length=getattr(self.config, "max_encoder_position_embeddings", None))
        from src.HDT import HDTTokenizer
        self.htrans_tokenizer = HDTTokenizer(self.tokenizer, max_document_length=max_len)
        self.use_ctrl_codes = use_ctrl_codes
        self.reqd_token_idx = 0 if not use_ctrl_codes else 1
        self._task_id = task_id
        if self._task_id:
            if use_ctrl_codes:
                logger.info(f"Control code used: {self._task_id}")
            elif variant != "default":
                logger.info(f"Task id used: {self._task_id}")
        self.use_fp16 = use_fp16

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        if self.use_ctrl_codes:
            logger.info(f"Control code used: {value}")
        elif self.variant != "default":
            logger.info(f"Task id used: {value}")
        self._task_id = value

    @torch.no_grad()
    def __call__(self, batch, batch_ids=None):
        def append_ctrl_code(batch, batch_ids):
            if type(self._task_id) == dict:
                batch = [f"{self.task_id['query']} {text}" if bid[1] == "q" else f"{self.task_id['candidates']} {text}"
                         for text, bid in zip(batch, batch_ids)]
            else:
                batch = [f"{self.task_id} {text}" for text in batch]
            return batch

        batch = [batch] if type(batch) == str else batch
        batch_ids = [] if not batch_ids else batch_ids
        if self.use_ctrl_codes:
            batch = append_ctrl_code(batch, batch_ids)
        inputs = []

        # pad_token_type_ids = np.zeros((1, self.config.max_sent_length), dtype=np.int64)

        if self.document:
            for sample in batch:
                if type(sample) == str:
                    sample = [[sample]]
                inputs.append(self.htrans_tokenizer(sample))
            input_ids = dict(zip(inputs[0].keys(),
                                 [torch.tensor(np.concatenate([np.expand_dims(d[key], axis=0) for d in inputs])) for key in inputs[0].keys()]))

        else:
            pad_input_ids = np.ones((1, self.config.max_sent_length), dtype=np.int64) * self.tokenizer.pad_token_id
            pad_attention_mask = np.zeros((1, self.config.max_sent_length), dtype=np.int64)
            for sample in batch:
                if type(sample) == str:
                    sample = [sample]
                else:
                    sample = list(chain.from_iterable(sample))
                sentences = sample[:self.config.max_sec_length]
                tokenized_sample = self.tokenizer(sentences, padding="max_length", truncation=True,
                                                  return_tensors="np", return_token_type_ids=False)
                inputs.append({"input_ids": np.row_stack([tokenized_sample["input_ids"]] + [pad_input_ids] * (
                            self.config.max_sec_length - len(sentences))).reshape(
                    (1, self.config.max_sent_length * self.config.max_sec_length)),
                               "attention_mask": np.row_stack(
                                   [tokenized_sample["attention_mask"]] + [pad_attention_mask] * (
                                               self.config.max_sec_length - len(sentences))).reshape(
                                   (1, self.config.max_sent_length * self.config.max_sec_length)),
                               "sec_mask": np.column_stack(
                                   [np.ones((1, tokenized_sample["input_ids"].shape[0]), dtype=np.int64)] + (
                                           self.config.max_sec_length - tokenized_sample["input_ids"].shape[0]) * [
                                       np.zeros((1, 1), dtype=np.int64)]),
                               "head_ids": np.array(
                                   [[self.tokenizer.get_vocab()["<sec>"], self.tokenizer.get_vocab()["<doc>"]]],
                                   dtype=np.int64)
                               })
            input_ids = dict(zip(inputs[0].keys(),
                                 [torch.tensor(np.concatenate([d[key] for d in inputs])) for key in inputs[0].keys()]))
        if "special_tokens_mask" in input_ids.keys():
            input_ids.pop("special_tokens_mask")
        input_ids = move_to_device(input_ids, self.encoder.device)

        if self.variant == "default":
            with torch.cuda.amp.autocast():
                if isinstance(self.encoder, BudgetModel):
                    output = self.encoder(input_ids, output_hidden_states=True)
                else:
                    input_ids["keep_ids"] = [input_ids.pop("keep_ids_0"), input_ids.pop("keep_ids_1"), input_ids.pop("keep_ids_2")]
                    input_ids["hash_ids"] = [input_ids.pop("hash_ids_0"), input_ids.pop("hash_ids_1"),
                                             input_ids.pop("hash_ids_2")]
                    input_ids["position_ids"] = [input_ids.pop("position_ids_0"), input_ids.pop("position_ids_1"),
                                             input_ids.pop("position_ids_2")]
                    output = self.encoder(**input_ids, output_hidden_states=True)
        elif type(self._task_id) != dict:
            output = self.encoder(task_id=self._task_id, **input_ids)
        else:
            x = input_ids["input_ids"]
            output = torch.zeros(x.shape[0], x.shape[1], self.config.hidden_size).to("cuda")
            q_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "q"])
            c_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "c"])

            if not q_idx.shape[0]:
                output = self.encoder(task_id=self._task_id["candidates"], **input_ids)
            elif not c_idx.shape[0]:
                output = self.encoder(task_id=self._task_id["query"], **input_ids)
            else:
                for i, v in enumerate(sorted(self._task_id.values())):
                    curr_input_idx = q_idx if v == "[QRY]" else c_idx
                    curr_input = x[curr_input_idx]
                    curr_output = self.encoder(task_id=v, input_ids=curr_input,
                                               attention_mask=input_ids["attention_mask"][curr_input_idx])
                    try:
                        output[curr_input_idx] = curr_output  # adapters
                    except:
                        output[curr_input_idx] = curr_output.last_hidden_state  # pals
        try:
            if hasattr(output, "pooler_output"):
                embedding = output.pooler_output
            elif self.sparse:
                embedding = output.hidden_states[-1][:, 0]
            elif self.config.pool_scheme == "first-token":
                # embedding = output.last_hidden_state[:, self.reqd_token_idx, :]  # cls token
                embedding = output.hidden_states[-3][:, [i * self.config.max_sent_length + self.reqd_token_idx for i in
                                                         range(self.config.max_sec_length)], :].mean(dim=1)  # cls token
            elif self.config.pool_scheme == "avg":
                # embedding = output.last_hidden_state.mean(dim=1)
                embedding = output.hidden_states[-3][:, [i * self.config.max_sent_length + self.reqd_token_idx for i in
                                                         range(self.config.max_sec_length)], :].mean(dim=1)  # cls token
                # embedding = output.last_hidden_state[:, self.reqd_token_idx, :]  # cls token
            elif self.config.pool_scheme == "max":
                embedding = output.hidden_states[-3].max(dim=1)[0]
        except:
            embedding = output.hidden_states[-1][:, self.reqd_token_idx, :]  # cls token
        return embedding.half() if self.use_fp16 else embedding


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


TASK_IDS = {"classification": "[CLF]", "regression": "[RGN]", "proximity": "[PRX]",
            "adhoc_search": {"query": "[QRY]", "candidates": "[PRX]"}}

class SciRepEval:

    def __init__(self, tasks_config: str = "super_scirep.jsonl", task_list: List[str] = None, save_dir=".",
                 task_formats: List[str] = None, batch_size: int = 32, htrans=False, document=False, cache_dir=None):
        tasks_dict = dict()
        task_by_formats = dict()
        with open(tasks_config, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                tasks_dict[d["name"]] = d
                if d["type"] not in task_by_formats:
                    task_by_formats[d["type"]] = []
                task_by_formats[d["type"]].append(d["name"])
        if not task_list and not task_formats:
            self.tasks = tasks_dict
        elif task_list:
            self.tasks = {k: tasks_dict[k] for k in task_list}
        elif task_formats:
            self.tasks = dict()
            for task_format in task_formats:
                self.tasks.update({k: tasks_dict[k] for k in task_by_formats[task_format]})
        self.batch_size = batch_size
        self.document = document
        self.htrans = htrans
        self.cache_dir = cache_dir
        self.save_dir = save_dir
    def evaluate(self, model):
        final_results = dict()
        if type(model) != list:
            model = [model]
        for task_name, task in tqdm(self.tasks.items(), total=len(self.tasks)):
            for m in model:
                m.task_id = TASK_IDS[task["type"]]
            kwargs = dict()
            task_data = task["data"]
            if not task_data.get("meta"):
                raise ValueError(f"Task {task_name} has no test metadata")
            if task_data.get("meta"):
                metadata = task_data["meta"]
                kwargs["meta_dataset"] = metadata if type(metadata) != dict else (metadata["name"], metadata["config"])

            if not task_data.get("test"):
                if type(metadata) == dict:
                    kwargs["test_dataset"] = (metadata["name"], metadata["config"])
                else:
                    raise ValueError(f"Task {task_name} has no test data")
            if task_data.get("test"):
                testdata = task_data["test"]
                kwargs["test_dataset"] = testdata if type(testdata) != dict else (testdata["name"], testdata["config"])

            kwargs["metrics"] = tuple(task["metrics"])
            kwargs["cache_dir"] = self.cache_dir
            kwargs["batch_size"] = task["batch_size"] if "batch_size" in task else self.batch_size

            if "fields" in task:
                kwargs["fields"] = task["fields"]
            save_path, load_path = None, None
            if "embeddings" in task:
                save_path = os.path.join(self.save_dir, task["embeddings"].get("save")) if task["embeddings"].get("save") is not None else task["embeddings"].get("save")
                load_path = os.path.join(self.save_dir, task["embeddings"].get("load")) if task["embeddings"].get("load") is not None else task["embeddings"].get("load")
            few_shot_evaluators = []
            if task["type"] in {"classification", "regression"}:
                subtype = SupervisedTask.CLASSIFICATION if task[
                                                               "type"] == "classification" else SupervisedTask.REGRESSION
                if task.get("multi_label"):
                    subtype = SupervisedTask.MULTILABEL_CLASSIFICATION
                evaluator = SupervisedEvaluator(task_name, subtype, model=model,
                                                **kwargs)
                if task.get("few_shot"):
                    for run in task["few_shot"]:
                        few_shot_evaluators.append(
                            FewShotEvaluator(f"{task_name} {run['sample_size']} shot", subtype, model=model,
                                             sample_size=run["sample_size"], num_iterations=run["iterations"],
                                             **kwargs))
            else:
                if task_name == "Paper-Reviewer Matching":
                    if not task_data.get("reviewers") and not task_data.get("hf_reviewers"):
                        raise ValueError(f"Task {task_name} has no reviewer metadata locally or hf_metadata")
                    if task_data.get("reviewers"):
                        reviewers = task_data["reviewers"]
                        kwargs["reviewer_metadata"] = reviewers if type(reviewers) != dict else (
                            reviewers["name"], reviewers["config"])
                    evaluator = ReviewerMatchingEvaluator(task_name, model=model, **kwargs)
                else:
                    data_class = SimpleDataset if task_data.get("simple_format") else IRDataset
                    evaluator = IREvaluator(task_name, model=model, dataset_class=data_class, **kwargs)
            embeddings = evaluator.generate_embeddings(save_path, htrans=self.htrans, document=self.document) if not load_path else load_path
            results = evaluator.evaluate(embeddings, result_save_dir=self.save_dir)
            if not few_shot_evaluators:
                final_results[task_name] = results
            else:
                final_results[task_name] = dict()
                final_results[task_name]["complete"] = results
                final_results[task_name]["few_shot"] = []

            for few_shot in few_shot_evaluators:
                final_results[task_name]["few_shot"].append(
                    {"sample_size": few_shot.sample_size, "results": few_shot.evaluate(embeddings)})
            # final_results[task_name]["task_name"] = task_name
        return final_results