from re import T
import time
import torch.nn as nn

out_dir = 'out-shakespeare'
eval_interval = 100
eval_iters = 10
wandb_log = True  # feel free to turn on
wandb_project = 'alpaca-nanoKAN'
wandb_run_name = 'ft-' + str(time.time())
dataset = 'alpaca'
init_from = 'gpt2'  # this is the largest GPT-2 model
always_save_checkpoint = False

batch_size = 1
gradient_accumulation_steps = 32
max_iters = 200000000
learning_rate = 3e-5
decay_lr = True 
learning_rate = 6e-4 # max learning rate


n_embd = 768
dropout = 0.1

kan_hidden_size = 768
grid_size = 5
spline_order = 3
scale_noise = 0.1
scale_base = 1.0
scale_spline = 1.0
base_activation = nn.SiLU
grid_eps = 0.02
grid_range = [-1, 1]


drop = True 
att_resid = True 
mlp_resid = True
ln = True


downsizeMLP = False 
downsize_size = 256