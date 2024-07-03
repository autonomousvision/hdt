import os
import torch
import time
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from src.lightning_modules.basics import BasicModel, BudgetModel
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from src.HDT import HDTEncoderModel, HDTEncoderForPreTraining, HDTConfig, HDTForConditionalGeneration
from typing import Optional, Any, List, Union
import configs as CONFIG
from src.utils import module_to_dict
from itertools import chain
from src.utils.metrics import Scrolls, F1_classification

def batch_preprocess(batch, pad_token_id):
    batch["keep_ids"] = [torch.tensor(batch.pop("keep_ids_0"), dtype=torch.int32),
                         torch.tensor(batch.pop("keep_ids_1"), dtype=torch.int32)] + (
                            [torch.tensor(batch.pop("keep_ids_2"),
                                          dtype=torch.int32)] if "keep_ids_2" in batch else [])
    batch["hash_ids"] = [torch.tensor(batch.pop("hash_ids_0"), dtype=torch.int32),
                         torch.tensor(batch.pop("hash_ids_1"), dtype=torch.int32)] + (
                            [torch.tensor(batch.pop("hash_ids_2"),
                                          dtype=torch.int32)] if "hash_ids_2" in batch else [])
    batch["position_ids"] = [batch.pop("position_ids_0"), batch.pop("position_ids_1")] + (
        [batch.pop("position_ids_2")] if "position_ids_2" in batch else [])
    batch.pop("special_tokens_mask", None)
    batch.pop("ids", None)
    batch.pop("questions", None)
    batch.pop("all_answers", None)
    batch["labels"] = torch.where(batch["labels"] == pad_token_id, -100, batch["labels"])
    return batch


class HDTFT(BasicModel):
    def __init__(self):
        super().__init__()
        config_class = HDTConfig
        if CONFIG.cfg_model.encoder_only:
            model_class = HDTEncoderModel
            self.metric = F1_classification(CONFIG.exps_config.num_labels, CONFIG.exps_config.task)
        else:
            model_class = HDTForConditionalGeneration
            self.metric = Scrolls(config_name=CONFIG.exps_config.task_type)
        self.encoder_only = CONFIG.cfg_model.encoder_only
        if CONFIG.pretrained_checkpoint:
            model_config = config_class.from_pretrained(CONFIG.pretrained_checkpoint)
            model_config.num_labels = CONFIG.exps_config.num_labels
            # model_config.problem_type = CONFIG.problem_type
            self.model = model_class.from_pretrained(CONFIG.pretrained_checkpoint, config=model_config)
        else:
            model_config = config_class(**module_to_dict(CONFIG.cfg_model))
            model_config.num_labels = CONFIG.exps_config.num_labels
            self.model = model_class(model_config)

    def forward(self, batch, *args: Any, **kwargs: Any) -> Any:
        batch = self.batch_preprocess(batch)
        if self.encoder_only:
            batch["decoder_attention_mask"] = batch.pop("answer_mask")
        return self.model(*args, **batch)

    def training_step(self, batch, batch_idx: int, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        outputs = self.forward(batch)
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx: int, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch = self.batch_preprocess(batch)
        outputs = self.model(*args, **batch)
        return outputs.logits, batch["labels"]


    def test_step(self, batch, batch_idx: int, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch = self.batch_preprocess(batch)
        outputs = self.model(*args, **batch)
        return outputs.logits, batch["labels"]

    def gneration_evaluation(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]], prefix="val") -> None:
        ids = list(chain.from_iterable([i["ids"] for i in outputs]))
        predictions = list(chain.from_iterable([i["preds"] for i in outputs]))
        references = list(chain.from_iterable([i["references"] for i in outputs]))
        metric_val = self.metric.compute(predictions=predictions, references=references)
        # self.log(f"{prefix}/ABS_F1", np.mean(abstractive_f1s))
        for score, key in zip(metric_val["display"], metric_val["display_keys"]):
            self.log(f"{prefix}_{key}", score)
        if np.mean(metric_val["display"]) > self.max_metric_val:
            self.max_metric_val = np.mean(metric_val["display"])
            self.model.save_pretrained(os.path.join(CONFIG.save_dir, "best_model"))
            with open(os.path.join(CONFIG.save_dir, "best_model", f"{prefix}_metric_val.json"), 'w') as f:
                json.dump(metric_val, f)
            pd.DataFrame({"id": ids, "prediction": predictions, "references": [i[0] for i in references]}).to_excel(
                os.path.join(CONFIG.save_dir, "best_model", f"{prefix}.xlsx"))

    def encoder_evaluation(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]], prefix="val") -> None:
        logits = torch.concat([i[0] for i in outputs], dim=0).detach().cpu()
        labels = torch.concat([i[1] for i in outputs], dim=0).detach().cpu()
        results = self.metric(logits, labels.squeeze())
        if isinstance(results, dict):
            for key, value in results.items():
                self.log(f"{prefix}/{key}", value)
        else:
            self.log(f"{prefix}/{self.cfg_model.metric}", results)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.evaluation(outputs)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.evaluation(outputs)


class HDTPretrain(BudgetModel):
    def __init__(self):
        super().__init__()
        if CONFIG.cfg_model.encoder_only:
            model_class = HDTEncoderForPreTraining
        else:
            model_class = HDTForConditionalGeneration
        self.encoder_only = CONFIG.cfg_model.encoder_only
        self.model = model_class(HDTConfig(**module_to_dict(CONFIG.cfg_model)))
        # Load the trained/constructed tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.save_dir)

    def forward(self, batch, *args: Any, **kwargs: Any) -> Any:
        batch = batch_preprocess(batch, self.tokenizer.pad_token_id)
        return self.model(**kwargs, **batch)

    def training_step(self, batch, batch_idx: int, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        if self.check_budget(self.wallclock_timer, self.budget, self.bias_timer):
            print(f'Signal stop found, stopping at epoch {self.trainer.current_epoch}')
            self.trainer.save_checkpoint(os.path.join(CONFIG.checkpoint_dir, "last.ckpt"))
            self.model.eval()
            self.model.save_pretrained(CONFIG.save_dir)
            self.tokenizer.save_pretrained(CONFIG.save_dir)
            self.trainer.should_stop = True
            self.trainer.strategy.barrier()
        if self.check_budget(self.evaluation_timer, self.budget / 3):
            self.model.eval()
            save_path = os.path.join(CONFIG.save_dir, "_".join(time.asctime().split(" ")))
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.model.train()
        outputs = self.forward(batch)
        # Here we plot the mlm_loss as train loss for comparison with other methods without permutation
        self.log("train/loss", outputs.loss)
        # self.log("train/mlm_loss", outputs.mlm_loss)
        return outputs

    def validation_step(self, batch, batch_idx: int, *args: Any, **kwargs: Any):
        outputs = self.forward(batch)
        # TODO: full text replace #ref9# to #reference#, '#eq1#' to #latex#
        self.log("val/loss", outputs.loss)

    def test_step(self, batch, batch_idx: int, *args: Any, **kwargs: Any):
        outputs = self.forward(batch)
        self.log("test/loss", outputs.loss)