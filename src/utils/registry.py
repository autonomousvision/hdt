optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "src.utils.optim.lamb.JITLamb",
    'adad':    "torch.optim.Adadelta",
    'adag':    "torch.optim.Adagrad",
    'adafactor': "src.models.optimizers.Adafactor",
    'deepspeed_adam': "deepspeed.ops.adam.DeepSpeedCPUAdam",
    'fused_adam': "deepspeed.ops.adam.FusedAdam"
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine":     "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

callbacks = {
    "timer":                 "src.callbacks.timer.Timer",
    "params":                "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint":      "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping":        "pytorch_lightning.callbacks.EarlyStopping",
    "swa":                   "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary":    "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar":     "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing":  "src.callbacks.progressive_resizing.ProgressiveResizing",
    # "profiler": "pytorch_lightning.profilers.PyTorchProfiler",
}

metric = {
    "kendal_tau": "scipy.stats.kendalltau",
    "multi_label_f1": "torchmetrics.functional.classification.multilabel_f1_score"
}

model = {
    # Backbones from this repo
    "bert_pretrain": "src.models.bert.BERTPretrain",
    "bert_ft": "src.models.bert.BERTFT",
    "hierarchical_pretrain": "src.models.hsformer.HierarchicalPretrain",
    "hierarchical_ft": "src.models.hsformer.HierarchicalFT",
    "hierarchical_v2_ft": "src.models.hsformerv2.HierarchicalV2FT",
    "hierarchical_v2_pretrain": "src.models.hsformerv2.HierarchicalV2Pretrain",
    "hierarchical_v3_ft": "src.models.hsformerv3.HierarchicalV3FT",
    "hierarchical_v3_pretrain": "src.models.hsformerv3.HierarchicalV3Pretrain",
    "hf_mlm_pretrain": "src.models.huggingface_model.HFMLMPretrain",
    "hf_seq2seq_pretrain": "src.models.huggingface_model.HFLMPretrain",
    "hf_ft": "src.models.huggingface_model.HfFT",
    "hed_pretrain": "src.models.hed.HEDPretrain",
    "hed_ft": "src.models.hed.HEDFT",
    "hf_qasper": "src.models.huggingface_model.HFQasper",
    "hed_scrolls_ft": "src.models.hed.HEDScrolls",
    "hf_scrolls": "src.models.huggingface_model.HFScrolls",
    "hed_conditionalQA_ft": "src.models.hed.model.HEDConditionalQA",
    "hed_encoder_ft": "src.models.hed.model.HierarchicalV3FT",
    "hat_ft": "src.models.huggingface_model.HATFT",
    "hat_triplet": "src.models.huggingface_model.HATTriplet",
    "hed_triplet": "src.models.hed.model.HierarchicalTriplet",
    "hf_triplet": "src.models.huggingface_model.HFTriplet",
    "hed_facetsum": "src.models.hed.model.HEDFacetSum",
    "hf_facetsum": "src.models.huggingface_model.HFFacet"
}

datamodule = {
    "mlm": "src.datamodules.datamodules.MLMDataModule",
    "ft": "src.datamodules.datamodules.FTDataModule",
    "hierarchical_mlm": "src.datamodules.datamodules.HierarchicalMLMDataModule",
    "hierarchical_ft": "src.datamodules.datamodules.HierarchicalFTDataModule",
    "seq2seq": "src.datamodules.datamodules.Seq2SeqDataModule",
    "qasper_ft": "src.datamodules.datamodules.QasperFTDataModule",
    "conditionalQA_ft": "src.datamodules.datamodules.ConditionalQAFTDataModule",
    "ul2": "src.datamodules.datamodules.UL2DataModule",
    "ul2_wo_cache": "src.datamodules.datamodules.UL2DataModuleNoCache",
    "mlm_wo_cache": "src.datamodules.datamodules.MLMDataModuleNoCache",
    "encoder_ft": "src.datamodules.datamodules.EncoderFTDataModule",
    "triplet_wo_cache": "src.datamodules.datamodules.TripletDataModuleNoCache"
}

preprocessor = {
    "hierarchical_sparse": "src.datamodules.preprocess.hierarchical_data_fn",
    "scirep": "src.datamodules.preprocess.scirep_data_fn",
    "math_bert": "src.datamodules.preprocess.math_bert_data_fn",
    "math_add_mul_add_bert": "src.datamodules.preprocess.math_bert_add_mul_add_data_fn",
    "sequence": "src.datamodules.preprocess.sequence_data_fn",
    "seq2seq": "src.datamodules.preprocess.seq2seq_data_fn",
    "qasper": "src.datamodules.preprocess.qasper_data_fn",
    "ul2": "src.datamodules.preprocess.ul2data_fn",
    "no_tokenize": "src.datamodules.preprocess.no_tokenize_data_fn",
    "scrolls": "src.datamodules.preprocess.scrolls_data_fn",
    "conditional_qa": "src.datamodules.preprocess.conditionalQA_data_fn",
    "encoder": "src.datamodules.preprocess.encoder_data_fn",
    "gov_report": "src.datamodules.preprocess.gov_report_data_fn",
    "triplet": "src.datamodules.preprocess.triplet_data_fn",
    "facetsum": "src.datamodules.preprocess.facetsum_data_fn"
}