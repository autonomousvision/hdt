import lightning as L
from argparse import ArgumentParser
from src.data import UL2DataModule, HierarchicalMLMDataModule
from src.lightning_modules import HDTPretrain
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.utils import module_to_dict
import configs as CONFIG


def main():
    if CONFIG.model_config.encoder_only:
        dataloader = HierarchicalMLMDataModule()
    else:
        dataloader = UL2DataModule()
    dataloader.prepare_data()
    CONFIG.cfg_model.vocab_size = len(dataloader.tokenizer)
    model = HDTPretrain()
    logger = WandbLogger(save_dir=CONFIG.save_dir, **module_to_dict(CONFIG.cfg_logger))

    trainer = Trainer(**module_to_dict(CONFIG.cfg_trainer), logger=logger, callbacks=[LearningRateMonitor(logging_interval='step')])
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    seed_everything(CONFIG.cfg_exps.seed, workers=True)
    main()