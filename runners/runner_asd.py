import sys
sys.path.append("./../")
from runners.base_runner import BaseRunner
import logging
import time
import torch
from utils.average_meter import AverageMeter
import os
from torch.nn import functional as F
from metrics import compute_auroc
import numpy as np
import math
from utils.savefig import savefig, savefig_landslide_detection, savefig_argriculture_vision
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import _LRScheduler
from numpy import ndarray as NDArray
from tqdm import tqdm
import time
import random
import sys
sys.path.append('./../')
from criterions.density import GaussianDensitySklearn, GaussianDensityTorch


class ASD_Runner(BaseRunner):

    def _train(self, epoch: int) -> None:
        self.model.train()
        train_iter_loss = AverageMeter()
        train_compact_loss = AverageMeter()
        train_diversity_loss = AverageMeter()
        train_recons_loss = AverageMeter()

        epoch_start_time = time.time()
        self.dists = []

        if self.first_epoch:
            logging.info("first epoch processing")
            self.R = torch.tensor(self.cfg.model.R, device=self.cfg.params.device)
            self.c  = self.init_center_c()
            self.optimizers = []
            self.optimizers.append(self.optimizer)
            self.schedulers = []
            self.schedulers.append(self.scheduler)
            self.scale_num = len(list(map(float, self.cfg.model.args.scales.split(', '))))
            self.loss_ratio = list(map(float, self.cfg.params.loss_ratio.split(', '))) # loss_ratio
            self.loss_ratio = torch.from_numpy(np.array(self.loss_ratio)).to(self.cfg.params.device)
            for i in range(self.scale_num):
                self.optimizers.append(self._re_init_optimizer(self.model.models[i].parameters()))
                self.schedulers.append(self._re_init_scheduler(self.optimizers[i]))

            self.train_loader_size = self.dataloaders['train'].__len__()
            self.first_epoch = False

        last_compact_loss = -1
        for batch_idx, (imgs, mask, transformed_imgs) in enumerate(self.dataloaders["train"]):
        
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            
            imgs = imgs.to(self.cfg.params.device)
            transformed_imgs = transformed_imgs.to(self.cfg.params.device)

            valid_locations = np.where(mask != 0)
            if len(valid_locations[1]) == 0:
                continue

            descriptors, recons_map = self.model(imgs)

            if self.cfg.datasets.train.name == 'datasets - AgricultureVisonDataset' and batch_idx > 500:
                break
        
            if epoch > 10:
                transformed_image_descriptors, _ = self.model(transformed_imgs)
                temp_c = self.c.reshape((1, self.cfg.params.latent_dim, 1, 1))
                temp_c = temp_c.repeat(imgs.shape[0], 1, self.cfg.params.image_size, self.cfg.params.image_size)
                valid_dists = torch.sum((descriptors - temp_c) ** 2, dim=1)[0, valid_locations[1], valid_locations[2]]
                self.dist = torch.max(valid_dists)
                self.dists.append(self.dist.item())
                scores = valid_dists - self.R**2  
                compact_loss = self.R ** 2 + (1 / self.cfg.params.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                 
                diversity_loss = torch.mean(torch.pow((descriptors - transformed_image_descriptors), 2)[0, :, valid_locations[1], valid_locations[2]])
                diversity_loss = 1/(torch.sum(diversity_loss) + 1e-3)

                recons_loss = self.criterions['MSE'](imgs[0, :, valid_locations[1], valid_locations[2]], recons_map[0, :, valid_locations[1], valid_locations[2]])
            else:
                compact_loss = torch.tensor(0.0, device=self.cfg.params.device)
                diversity_loss = torch.tensor(0.0, device=self.cfg.params.device)
                recons_loss = self.criterions['MSE'](imgs[0, :, valid_locations[1], valid_locations[2]], recons_map[0, :, valid_locations[1], valid_locations[2]])
  
            loss = self.loss_ratio[0]*compact_loss + self.loss_ratio[1]*diversity_loss + self.loss_ratio[2]*recons_loss

            loss.backward()

            for optimizer in self.optimizers:
                optimizer.step()

            train_iter_loss.update(loss.item())
            train_compact_loss.update(compact_loss.item())
            train_recons_loss.update(recons_loss.item())
            train_diversity_loss.update(diversity_loss.item())


            if batch_idx % self.cfg.params.warm_up_n_iters == 0:
                spend_time = time.time() - epoch_start_time
                logging.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} total_loss:{:.6f} compact_loss:{:.6f} diversity_loss:{:.6f} recons_map:{:.6f} ETA:{}min'.format(epoch, batch_idx, self.train_loader_size, 
                batch_idx / self.train_loader_size * 100, self.optimizers[0].param_groups[-1]['lr'], train_iter_loss.avg, train_compact_loss.avg, train_diversity_loss.avg, train_recons_loss.avg, spend_time / (batch_idx + 1) * self.train_loader_size // 60 - spend_time // 60))
                
                if train_compact_loss.avg == last_compact_loss:
                    logging.info("compact_avg_loss: " + str(train_compact_loss.avg))
                    logging.info("last_compact_loss: " + str(last_compact_loss))
                    self.update_radius(epoch)
                    last_compact_loss = train_compact_loss.avg.clone()

                train_iter_loss.reset()
                train_compact_loss.reset()
                train_diversity_loss.reset()
                train_recons_loss.reset()
                
        self.update_radius(epoch)
        if self.cfg.params.change_center:
            self.c = self.init_center_c()

    def update_radius(self, epoch):
        new_R= torch.tensor(self.get_radius(), device=self.cfg.params.device)
        if not torch.isnan(new_R):
            self.R.data = new_R
        self.dists = []
        logging.info("new radius: " + str(self.R.data))

    def _re_init_optimizer(self, params) -> Optimizer:
        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=params)

    def _re_init_scheduler(self, optimizer) -> _LRScheduler:

        cfg = self.cfg.scheduler
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), optimizer=optimizer)


    def init_center_c(self, ):
        
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(self.cfg.params.latent_dim, device=self.cfg.params.device)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, _, _) in enumerate(tqdm(self.dataloaders["train"])):
                # get the inputs of the batch
                if batch_idx > 2000:
                    logging.warning(str(2000) + " images are used to compute the center!")
                    break

                inputs = imgs.to(self.cfg.params.device)
                descriptors, _,= self.model(inputs)
                n_samples += descriptors.shape[0]
                c += torch.sum(torch.mean(descriptors, dim=(2,3)), dim=0)

        c /= n_samples
        logging.info("c is set with " + str(c))

        return c

    def get_radius(self, ):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.mean(np.sqrt(np.array(self.dists)))
     

    def get_embeddings(self, total_embedding_num = 1000.0):
        
        self.train_loader_size = self.dataloaders['train'].__len__() 

        valid_image_num = min(2000, self.train_loader_size)
        embedding_num_per_image = int(total_embedding_num / valid_image_num)
        embeddings = []

        count = 0
        with torch.no_grad():
            for batch_idx, (imgs, mask, _) in enumerate(tqdm(self.dataloaders["train"])):
                if count > valid_image_num:
                    logging.warning(str(count) + " images are used to get the normal embeddings in the training set")
                    break
                count += 1
                # get the inputs of the batch
                inputs = imgs.to(self.cfg.params.device)
                descriptors, _, = self.model(inputs)
                valid_locations = np.where(mask != 0)
                descriptors = descriptors.permute((0,2,3,1))
                descriptors = descriptors[0, valid_locations[1], valid_locations[2], :]
                num = descriptors.shape[0]
                descriptors = descriptors.detach().cpu().numpy()
                li=list(range(num))
                random.shuffle(li) 
                embeddings += descriptors[li[0:max(embedding_num_per_image, len(valid_locations[1]))],: ].tolist()

        embeddings = np.array(embeddings)
        return (embeddings) 


    def _test(self, epoch: int, visualization=False) -> None:

        self.model.eval()
        embeddings = self.get_embeddings(self.cfg.params.embedding_num)
        embeddings = torch.from_numpy(embeddings).to(self.cfg.params.device)
        gde = GaussianDensityTorch()
        gde.fit(embeddings, self.cfg.params.device) 

        artifacts: Dict[str, List[NDArray]] = {
            "img": [],
            "gt": [],
            "amap": [],
        }
        
        count = 0
        for mb_img, mb_gt in tqdm(self.dataloaders["test"]):

            if len(self.dataloaders["test"]) > 10000 and count > 2000:
               logging.warning(str(count) + " images are used to be evaluated")
               break

            count += 1

            if self.cfg.datasets.train.name == 'datasets - AgricultureVisonDataset' and count > 2000:
                break

            if visualization and mb_gt.max() == 2:
                continue
            
            mb_amap = 0
            with torch.no_grad():
                mb_img = mb_img.to(self.cfg.params.device)
                descriptors, _ = self.model(mb_img)
        
                feature_shape = descriptors.shape
                descriptors = descriptors.permute((0,2,3,1)).reshape((feature_shape[0]*feature_shape[2]*feature_shape[3], feature_shape[1]))
                anomaly_map = gde.predict(descriptors)
                anomaly_map = anomaly_map.reshape(1, feature_shape[2], feature_shape[3])
                mb_amap += anomaly_map.unsqueeze(1)
                    
            artifacts["amap"].extend(mb_amap.squeeze(1).detach().cpu().numpy())
            artifacts["img"].extend(mb_img.permute(0, 2, 3, 1).detach().cpu().numpy())
            artifacts["gt"].extend(mb_gt.detach().cpu().numpy())


        artifacts["amap"] = np.array(artifacts["amap"])
        artifacts["amap"] = 1 - (artifacts["amap"] - artifacts["amap"].min())/(artifacts["amap"].max() - artifacts["amap"].min())

        try:
            auroc = compute_auroc(epoch, np.array(artifacts["amap"]), np.array(artifacts["gt"]), self.working_dir, image_level=False, compute_iou=True)
        except ValueError:
            logging.info('Exception happens')
            pass 

        ep_amap = np.array(artifacts["amap"])
        ep_amap = 1 - ep_amap
        artifacts["amap"] = list(ep_amap)

        ep_gt = np.array(artifacts["gt"])
        ep_gt = 1 - ep_gt
        artifacts["gt"] = list(ep_gt)
        
        if visualization:
            savefig(epoch, artifacts["img"],  artifacts["gt"], artifacts["amap"], self.working_dir, self.mean, self.std, 255)  # save results on DeepGlobe
            savefig(epoch, artifacts["img"],  artifacts["gt"], artifacts["amap"], self.working_dir, 1, 1, 255)  # save results on FAS
            # savefig_argriculture_vision(epoch, artifacts["img"],  artifacts["gt"], artifacts["amap"], self.working_dir, self.mean, self.std, 255) # save results on Agriculture-Vision
            # savefig_landslide_detection(epoch, artifacts["img"],  artifacts["gt"], artifacts["amap"], self.working_dir, self.mean, self.std, 255) # save results on Landslide4Sense


    def save(self, epoch, save_best=False):
        self.scales = list(map(float, self.cfg.model.args.scales.split(', ')))
        self.scale_num = len(self.scales)
        for i in range(self.scale_num):
            if save_best:
                save_path = os.path.join(self.working_dir, "epochs-best-" + str(self.R.data.item()))
                self.best_save_dir = save_path
            else:
                save_path = os.path.join(self.working_dir, "epochs-" + str(epoch))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.join(save_path, 'checkpoint-model-scale-{}.pth'.format((str(self.scales[i]) + str(i))))
            torch.save(self.model.models[i].state_dict(), filename)
        filename = os.path.join(save_path, 'checkpoint-model-{}.pth'.format('mseoad'))
        torch.save(self.model.state_dict(), filename)


    def load(self, save_dir):
        self.scales = list(map(float, self.cfg.model.args.scales.split(', ')))
        self.scale_num = len(self.scales)
        for i in range(self.scale_num):
            filename = os.path.join(save_dir, 'checkpoint-model-scale-{}.pth'.format((str(self.scales[i]) + str(i))))
            self.model.models[i].load_state_dict(torch.load(filename, map_location=torch.device(self.cfg.params.device)))

        filename = os.path.join(save_dir, 'checkpoint-model-{}.pth'.format('mseoad'))
        self.model.load_state_dict(torch.load(filename, map_location=torch.device(self.cfg.params.device)), strict=False)