import time
from torch.optim.lr_scheduler import LambdaLR


def _get_fake_step(current_step, initial_time, hour_budget, num_training_steps):
    elapsed_hours = (time.time() - initial_time) / 60 / 60
    if current_step == 0:
        fake_step = 0
    else:
        fake_step = int(elapsed_hours / hour_budget * num_training_steps)
    # avoid fake_step > num_training_steps
    return fake_step % num_training_steps


def get_budget_inv_sqrt_scheduler(optimizer, hour_budget, num_warmup_steps, num_cooldown_steps, num_training_steps):
    """Time-based scheduler as described in Iszak et al. plus inv_sqrt.
    Takes in num_warmup_steps and num_training_steps as normal, but actually squeezes the planned schedule into the
    budget given by hour_budget, based on wallclock measurements.

    Reference: https://github.com/IntelLabs/academic-budget-bert/blob/main/pretraining/schedules.py
    """
    decay_factor = num_warmup_steps**0.5
    decayed_lr = decay_factor * (num_training_steps - num_cooldown_steps) ** -0.5
    initial_time = time.time()

    def lr_lambda(current_step: int):
        fake_step = _get_fake_step(current_step, initial_time, hour_budget, num_training_steps)
        if fake_step < num_warmup_steps:
            return float(fake_step / num_warmup_steps)
        elif fake_step > (num_training_steps - num_cooldown_steps):
            return max(0.0, float(decayed_lr * (num_training_steps - fake_step) / num_cooldown_steps))
        else:
            return float(decay_factor * fake_step**-0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)