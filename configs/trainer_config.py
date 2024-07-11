def _set_args(num_gpus, accu_grad):
    global devices, strategy, accumulate_grad_batches
    if num_gpus > 1:
        accelerator = "ddp"
    devices = num_gpus
    accumulate_grad_batches = accu_grad

strategy = "auto"
min_epochs = 0 # prevents early stopping
max_epochs = 10
devices = 1
accelerator = "gpu"
# mixed precision for extra speed-up
precision = 16
profiler = "simple"
# perform a validation loop every N training epochs
check_val_every_n_epoch = 1
gradient_clip_val = 0.5
num_sanity_val_steps = 0
limit_val_batches = 0.0 # Disable validation for pre-training
val_check_interval = 1.0
# set True to to ensure deterministic results
# makes training slower but gives more reproducibility than just setting seeds
deterministic = True
accumulate_grad_batches = 16
# Disable validation for pre-training
auto_scale_batch_size = "power"