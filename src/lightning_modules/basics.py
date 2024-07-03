import time
import hydra
from pytorch_lightning import LightningModule
from src.utils.schedulers import get_budget_inv_sqrt_scheduler
from src.utils import registry, module_to_dict
import configs as CONFIG


class BasicModel(LightningModule):

    def __init__(self):
        super(BasicModel, self).__init__()

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": CONFIG.exps_config.optimizer_hyperparams["weight_decay"],
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = hydra.utils.get_method(path=registry.optimizer[CONFIG.exps_config.optimizer_name])(optimizer_grouped_parameters, lr=CONFIG.exps_config.lr, **CONFIG.exps_config.optimizer_hyperparams)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": hydra.utils.get_method(path=registry.scheduler[CONFIG.exps_config.scheduler_name])(optimizer, num_warmup_steps=0.1 * self.trainer.estimated_stepping_batches, num_training_steps=self.trainer.estimated_stepping_batches),
                "frequency": CONFIG.exps_config.scheduler_frequency,
                "interval": CONFIG.exps_config.scheduler_interval,
                "monitor": CONFIG.exps_config.scheduler_monitor,
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "Learning_Rate",
            },
        }


class BudgetModel(BasicModel):
    def __init__(self):
        super().__init__()
        self.budget = CONFIG.exps_config.budget
        self.wallclock_timer = time.time()
        self.evaluation_timer = time.time()
        self.bias_timer = 0
        self.cut_frac = CONFIG.exps_config.scheduer_cut_frac
    @staticmethod
    def check_budget(launch_time, hour_limit, bias=0):
        if hour_limit is not None:
            """These measurements are deliberately wall-clock based."""
            current_time = time.time()
            return True if (current_time - bias - launch_time) / 3600 > hour_limit else False
        else:
            return False

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": CONFIG.exps_config.optimizer_hyperparams["weight_decay"],
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = hydra.utils.get_method(path=registry.optimizer[CONFIG.exps_config.optimizer_name])(optimizer_grouped_parameters,
                                                                                           lr=CONFIG.exps_config.lr,
                                                                                           **CONFIG.exps_config.optimizer_hyperparams)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_budget_inv_sqrt_scheduler(optimizer, self.budget, self.trainer.estimated_stepping_batches * self.cut_frac, 0, self.trainer.estimated_stepping_batches),
                "frequency": CONFIG.exps_config.scheduler_frequency,
                "interval": CONFIG.exps_config.scheduler_interval,
                "monitor": CONFIG.exps_config.scheduler_monitor,
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "Learning_Rate",
            },
        }