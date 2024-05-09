import os
import subprocess
import wandb

def update_config_file(params, run_id, filename):
    """Update the configuration file with new parameter values."""
    unique_run_name = f"ft-{run_id}"
    unique_out_dir = f"out-alpaca-{run_id}"
    
    with open(filename, 'w') as f:
        f.write("from re import T\n")
        f.write("import time\n")
        f.write("import torch.nn as nn\n")
        f.write(f"out_dir = '{unique_out_dir}'\n")
        f.write(f"eval_interval = 100\n")
        f.write(f"eval_iters = 40\n")
        f.write(f"wandb_log = True\n")
        f.write(f"wandb_project = 'alpaca-gpt2'\n")
        f.write(f"wandb_run_name = '{unique_run_name}'\n")
        f.write(f"dataset = 'alpaca'\n")
        f.write(f"init_from = 'gpt2'\n")
        f.write(f"always_save_checkpoint = False\n")
        f.write(f"batch_size = 1\n")
        f.write(f"gradient_accumulation_steps = 32\n")
        f.write(f"max_iters = 2000\n")
        f.write(f"learning_rate = {params['learning_rate']}\n")
        f.write(f"decay_lr = True\n")
        f.write(f"n_embd = 768\n")
        f.write(f"dropout = 0.1\n")
        f.write(f"kan_hidden_size = {params['kan_hidden_size']}\n")
        f.write(f"grid_size = {params['grid_size']}\n")
        f.write(f"spline_order = {params['spline_order']}\n")
        f.write(f"scale_noise = {params['scale_noise']}\n")
        f.write(f"grid_eps = {params['grid_eps']}\n")
        f.write(f"scale_base = 1.0\n")
        f.write(f"scale_spline = 1.0\n")
        f.write(f"base_activation = nn.SiLU\n")
        f.write(f"grid_range = [-1, 1]\n")
        f.write(f"drop = True\n")
        f.write(f"att_resid = True\n")
        f.write(f"mlp_resid = True\n")
        f.write(f"ln = True\n")
        f.write(f"downsizeMLP = {params['downsizeMLP']}\n")
        f.write(f"downsize_size = {params['downsize_size']}\n")
        f.flush()
        f.close()
    # # os.fsync(f.fileno())
    #     f.close()
    #     f.flush()
def train():
    """Runs the training script with the updated configuration."""
    try: 
        run = wandb.init(dir='/root/wandbout')
        config_filename = f'config_{run.id}.py'
        update_config_file(run.config, run.id, config_filename)
        # Execute the training script with the specific config file
        subprocess.run(['python', 'train.py', config_filename])
        run.finish()
        # Optionally, clean up the config file after the run
        # os.remove(config_filename)
    except Exception as e:
        print(str(e))
def main():
    wandb.login()
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-5, 3e-5, 6e-5, 1e-4]
            },
            'grid_eps': {
                'values': [0.01, 0.02, 0.05]
            },
            'scale_noise': {
                'values': [0.05, 0.1, 0.15]
            },
            'grid_size': {
                'values': [3, 5, 7]
            },
            'spline_order': {
                'values': [2, 3, 4]
            },
            'downsize_size': {
                'values': [512,256,128,64]
            },
            'downsizeMLP': {
                'values': [True, False]
            },
            'kan_hidden_size': {
                'values': [
                    768, 512, 256, 128
                ]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='alpaca-exp')
    wandb.agent(sweep_id, train)

if __name__ == "__main__":
    main()
