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