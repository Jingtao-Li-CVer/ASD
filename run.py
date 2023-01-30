import hydra
import sys
import os 
from omegaconf import DictConfig
from utils.logging_ import logger_config
from importlib import import_module
from pathlib import Path
import logging
import torch
from utils.random_seed import setup_seed
import warnings 
import torch.multiprocessing as mp 

warnings.filterwarnings('ignore')
 
config_path = sys.argv[1]  
sys.argv.pop(1)


@hydra.main(config_path)
def main(cfg: DictConfig) -> None:
    
    setup_seed(cfg.random_seed)
    torch.autograd.set_detect_anomaly(True)
    
    working_dir = str(Path.cwd())
    logging.info(f"The current working directory is {working_dir}")
    
    runner_module_cfg = cfg.runner_module
    module_path, attr_name = runner_module_cfg.split(" - ")
    module = import_module(module_path)
    runner_module = getattr(module, attr_name)
    runner = runner_module(cfg, working_dir)

    runner.run()


if __name__ == "__main__":
    main() 