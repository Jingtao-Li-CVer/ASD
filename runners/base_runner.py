from abc import ABC, abstractmethod
from distutils.log import info
import logging
from albumentations import Compose
import hydra
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from importlib import import_module
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from typing import Any
import albumentations as A
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseRunner(ABC):
    def __init__(self, cfg: DictConfig, working_dir: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.transforms = {k: self._init_transforms(k) for k in self.cfg.transforms.keys()}
        self.first_epoch = True

        self.model = self._init_model()
        self.model = self.model.to(self.cfg.params.device)
        if self.cfg.model.ckpt_dir != '':
            self.load(self.cfg.model.ckpt_dir)
            logging.info("loading pretrained model......")
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        self.datasets = {k: self._init_datasets(k) for k in self.cfg.datasets.keys()}
        self.dataloaders = {k: self._init_dataloaders(k) for k in self.cfg.dataloaders.keys()}
        self.criterions = {k: self._init_criterions(k) for k in self.cfg.criterions.keys()}
        self.working_dir = working_dir

    def _init_transforms(self, key: str) -> Compose:

        transforms = []
        for cfg in self.cfg.transforms[key]:
            attr = self._get_attr(cfg.name)
            if cfg.name == 'albumentations - Normalize':
                self.mean = list(map(float, cfg.args.mean.split(', ')))
                self.std = list(map(float, cfg.args.std.split(', ')))
                transforms.append(A.Normalize(mean = self.mean, std = self.std, max_pixel_value=1.0))
            elif cfg.name == 'albumentations - ToFloat':
                self.max_value = list(map(float, str(cfg.args.max_value).split(', '))) 
                transforms.append(A.ToFloat(max_value=self.max_value)) 
            else:
                transforms.append(attr(**cfg.get("args", {})))
        return Compose(transforms)
    
    def _init_criterions(self, key: str) -> Module:

        cfg = self.cfg.criterions[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_datasets(self, key: str) -> Dataset:

        cfg = self.cfg.datasets[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), transforms=self.transforms[key])

    def _init_dataloaders(self, key: str) -> DataLoader:

        cfg = self.cfg.dataloaders[key]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.datasets[key])

    def cleanup(self, ):
        dist.destroy_process_group()

    def run(self) -> None:
        pbar = tqdm(range(1, self.cfg.params.epochs + 1), desc="epochs")
        self.model.to(self.cfg.params.device)
 
        for epoch in pbar:
            self._train(epoch)

            if self.cfg.runner_module == 'runners - ASD_Runner':
                for scheduler in self.schedulers:
                    scheduler.step(epoch)

            if epoch >= 10:
                save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                self.save(epoch)
                self._test(epoch, visualization=False)
                    

    def _init_model(self) -> Module:

        cfg = self.cfg.model
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=self.model.parameters())

    def _init_scheduler(self) -> _LRScheduler:

        cfg = self.cfg.scheduler
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), optimizer=self.optimizer)

    @abstractmethod
    def _train(self, epoch: int) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _test(self, epoch: int) -> None:

        raise NotImplementedError()

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)
