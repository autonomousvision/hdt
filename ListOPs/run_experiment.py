from huggingface_model import huggingface_model
from ExperimentRunnerArgs import ExperimentRunnerArgs
from model_save_results import save_results
from ListOpsDataModule import ListOPsDataModule
from pytorch_lightning import Trainer
import Config as Config
from pytorch_lightning.callbacks import ModelCheckpoint


# from src.SyntheticDataCreation.ListOps.create import create_dataset
from create_listops import create_dataset

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("operators", action="store", type=str)
    # parser.add_argument("max_digit", action="store", type=int)
    # parser.add_argument("tree_depth", action="store", type=int)
    # parser.add_argument("node_degree", action="store", type=int)
    parser.add_argument("value_p", action="store", type=float)
    parser.add_argument("max_args", action="store", type=int)
    parser.add_argument("tree_depth", action="store", type=int)
    parser.add_argument("train_size", action="store", type=int)
    parser.add_argument("n_layers", action="store", type=int)
    parser.add_argument("hidden_size", action="store", type=int) # hidden size = embedding dimension
    parser.add_argument("n_heads", action="store", type=int)
    parser.add_argument("intermediate_size", action="store", type=int)
    parser.add_argument("epochs", action="store", type=int)
    parser.add_argument("batch_size", action="store", type=int)
    parser.add_argument("learning_rate", action="store", type=float)
    parser.add_argument("scheduler", action="store", type=str)#
    parser.add_argument("hdt_attn", action="store", type=str)#
    parser.add_argument("longformer_local_attn_size", action="store", type=int)#
    parser.add_argument("model", action="store", type=str)
    parser.add_argument("results_savepath", action="store", type=str)

    args = parser.parse_args()
    return args

def main(args):
    print(args)
    dataset_name = create_dataset(args.value_p,
                                    args.max_args,
                                    args.tree_depth,
                                    args.train_size,
                                    test_size=10000)
    Config.set_learning_rate(args.learning_rate, args.scheduler)
    Config.set_data_paths(dataset_name)
    Config.set_epochs(args.epochs)
    Config.set_batch_size(args.batch_size)
    attn_color = 'full' if args.model == 'BERT' else args.hdt_attn 
    data_module = ListOPsDataModule()

    kwargs_list = [value for key,value in args._get_kwargs()]

    log_dir = '_'.join([str(arg) for arg in kwargs_list]) # untested line, not sure if args have keys
    import datetime
    log_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + log_dir 
    # Create the model
    if args.model == 'BERT':
        model = huggingface_model(model_name='BERT',
                                hidden_size=args.hidden_size, 
                               n_layers=args.n_layers, 
                               n_heads=args.n_heads,
                               intermediate_size=args.intermediate_size,
                               log_dir=log_dir)
    elif args.model == 'HDT':
        model = huggingface_model(model_name='HDT',
                    hidden_size=args.hidden_size, 
                    n_layers=args.n_layers,
                    n_heads=args.n_heads, 
                    intermediate_size=args.intermediate_size, 
                    attn_color=attn_color,
                    log_dir=log_dir)
    elif args.model == 'Longformer':
        model = huggingface_model(model_name='Longformer', 
                           hidden_size=args.hidden_size, 
                            n_layers=args.n_layers, 
                            n_heads=args.n_heads, 
                            intermediate_size=args.intermediate_size, 
                            attention_window=args.longformer_local_attn_size,
                            log_dir=log_dir)
    elif args.model == 'HAT':
        model = huggingface_model(model_name='HAT', 
                           hidden_size=args.hidden_size, 
                            n_layers=args.n_layers, 
                            n_heads=args.n_heads, 
                            intermediate_size=args.intermediate_size, 
                            log_dir=log_dir)
    else:
        raise ValueError(f'Unknown model: {args.model}')
 
    model_dataset_name = f'{args.model}_{args.n_layers}_{args.intermediate_size}_{args.epochs}_{dataset_name}'
    # Create the trainer
    import os
    trainer = Trainer(max_epochs=Config.EPOCHS, accelerator='gpu', devices=1,
                      callbacks=[# EarlyStopping(monitor='validation_accuracy', min_delta=0.00, patience=10, verbose=False, mode="max"),
                                 ModelCheckpoint(monitor='validation_accuracy', dirpath=os.path.join('runs', log_dir), filename=model_dataset_name, save_weights_only=True)],)

    # Train the model
    trainer.fit(model=model, datamodule=data_module)
    test_metrics = trainer.test(model=model, datamodule=data_module, verbose=True)[0]
    
    save_results(save_path = args.results_savepath, **test_metrics)


if __name__ == '__main__':
    import sys
    if len( sys.argv ) == 1: # No cmd line args were passed, using the debug configuration
        args = ExperimentRunnerArgs(
            value_p = 0.25,
            max_args = 5,
            tree_depth = 20,
            train_size = 90000,
            n_layers = 12,
            n_heads = 1,
            hidden_size = 128,
            intermediate_size = 123,
            epochs = 40,
            batch_size = 60,
            learning_rate = 0.0003,
            scheduler = 'fixed',
            hdt_attn = 'blue',
            model = 'HDT',
            results_savepath = '/results/hdt_test_run.json'
        )
    else:
        args = parse_args()
    main(args)
    