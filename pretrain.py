import argparse
from src.data import UL2DataModule, HierarchicalMLMDataModule
from src.lightning_modules import HDTPretrain
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.utils import module_to_dict
import configs as CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="logs/test", help="Directory to save the intermediate checkpoints and final model weights")
    parser.add_argument("--num_encoder_layers", default=6, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--cache_dir", default="cache", help="Path to save downloaded data/model cacje")
    parser.add_argument("--tok_name", default="google-t5/t5-base", help="Initialize with a trained tokenizer")
    parser.add_argument("--max_input_length", default=8192, type=int, help="Maximum input context length")
    parser.add_argument("--max_output_length", default=256, type=int, help="Maximum output context length (only valid for encoder-decoder model)")
    parser.add_argument("--mlm_probability", default=0.15, type=float)
    parser.add_argument("--lr", default=1e-3, help="Learning Rate")
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--budget", default=24, type=float, help="Number of hours the pre-training continues")
    parser.add_argument("--encoder_only", default=False, action="store_true", help="Encoder-only model or encoder-decoder model")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--accumulate_grad_batches", default=16, type=int, help="Number of batches for gradient accumulation")
    args = parser.parse_args()
    CONFIG.set_args(args)
    return args


def main():
    args = parse_args()
    if CONFIG.model_config.encoder_only:
        dataloader = HierarchicalMLMDataModule()
    else:
        dataloader = UL2DataModule()
    dataloader.prepare_data()
    CONFIG.cfg_model.vocab_size = len(dataloader.tokenizer)
    model = HDTPretrain(vocab_size=len(dataloader.tokenizer))
    logger = WandbLogger(save_dir=CONFIG.save_dir, **module_to_dict(CONFIG.cfg_logger))

    trainer = Trainer(**module_to_dict(CONFIG.cfg_trainer), logger=logger, callbacks=[LearningRateMonitor(logging_interval='step')])
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    seed_everything(CONFIG.cfg_exps.seed, workers=True)
    main()