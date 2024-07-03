import configs.data_config as cfg_data
import configs.model_config as cfg_model
import configs.exps_config as cfg_exps
import configs.trainer_config as cfg_trainer
import configs.logger_config as cfg_logger

hierarchical = True
save_dir = "logs/test"
cache_dir = "/home/haoyu/code/academic-budget-LMs/data/cache"
checkpoint_dir = save_dir
pretrained_checkpoint = False # either path to a directory containing pre-trained weights, or False
