def _set_args(learning_rate, B, T):
    global lr, batch_size, budget
    lr = learning_rate
    batch_size = B
    budget = T


num_labels = 1
lr = 1e-3
optimizer_name = "adamw"
optimizer_hyperparams = {"weight_decay": 0.01, "eps": 1e-8}
batch_size = 2
seed = 123
scheduler_name = "cosine_warmup"
scheduler_frequency = 1
scheduler_interval = "step"
scheduler_monitor = "train_loss"
scheduer_cut_frac = 0.5
budget = 24 #h
num_gpus = 1
task_type = "multiclass" # Or multilabel, only useful for classification tasks