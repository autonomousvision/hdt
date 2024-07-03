# hdt
[COLM'24] HDT: Hierarchical Document Transformer


## ListOPs experiments
We start experiments on [ListOPs](https://arxiv.org/abs/1804.06028) using the files in `/ListOPs`. 
The entry point is `run_experiment.py`. You can provide model names and hyperparameters as command line arguments. 
For example, to run the HDT vs BERT vs HAT vs Longformer experiment we used in the paper:  
`pip install -r requirements.txt`  
`cd ListOPs`    
`python run_experiment.py 0.25 5 20 90000 12 128 1 512 300 120 0.0003 fixed blue 512 HDT hdt_testrun`  
`python run_experiment.py 0.25 5 20 90000 12 128 1 512 300 120 0.0003 fixed blue 512 BERT bert_testrun`  
`python run_experiment.py 0.25 5 20 90000 12 128 1 512 300 120 0.0003 fixed blue 512 Longformer Longformer_testrun`  
`python run_experiment.py 0.25 5 20 90000 12 128 1 512 300 120 0.0003 fixed blue 512 HAT HAT_testrun`  