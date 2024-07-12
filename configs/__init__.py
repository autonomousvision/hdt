import configs.data_config as cfg_data
import configs.model_config as cfg_model
import configs.exps_config as cfg_exps
import configs.trainer_config as cfg_trainer
import configs.logger_config as cfg_logger


def set_args(args):
    global save_dir, cache_dir, checkpoint_dir
    save_dir = args.save_dir
    cache_dir = args.cache_dir
    checkpoint_dir = args.save_dir
    cfg_data._set_args(args.tok_name, args.max_input_length, args.mlm_probability)
    cfg_model._set_args(args.encoder_only, args.max_input_length, args.max_output_length, args.num_encoder_layers,
                       args.num_decoder_layers)
    cfg_exps._set_args(args.lr, args.batch_size, args.budget)
    cfg_trainer._set_args(args.num_gpus, args.accumulate_grad_batches)
    cfg_logger._set_args(f"HDT_{args.max_input_length}_{args.max_output_length}_encoder_{'' if args.encoder_only else 'decoder'}")



hierarchical = True # Always True before we have the code to pre-train a non-hierarchical model such as Longformer
save_dir = "lightning_logs/test"
cache_dir = "data"
checkpoint_dir = save_dir
pretrained_checkpoint = False # either path to a directory containing pre-trained weights, or False
