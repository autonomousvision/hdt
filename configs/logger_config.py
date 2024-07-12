def _set_args(run_name):
    global name
    name = run_name

offline = False
id = None  # pass correct id to resume experiment!
anonymous = None  # enable anonymous logging
project = "pretrain_HDT"
log_model: False  # upload lightning ckpts
prefix = ""  # a string to put at the beginning of metric keys
# entity: "" # set to name of your wandb team
group = ""
tags = []
job_type = ""
name = "HDT_8192"