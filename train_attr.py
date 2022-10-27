import argparse
import datetime
import json
import os
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp

# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import losses
import models
from dataset.dataset_dir import DatasetDir,DatasetImage
from utils import AverageMeter, MetaData
import matplotlib.pyplot as plt
import pickle as pkl
import shutil
from collections import defaultdict
import warnings
warnings.simplefilter("ignore", UserWarning)


cv2.setNumThreads(0)

import wandb

wandb.login()

dbg=False
num_attr=6
dbg_path='/media/dereyly/ssd4/ImageDB/xix/debug_attr_auto'

def unroll_prob(target, epoch=None):
    # target_out=torch.zeros_like(target)
    # rand = torch.randn_like(target)
    target_in=target.clone()
    if np.random.rand()>epoch/15 or np.random.rand()<0.02:
        rand = 0.05+0.75 * torch.rand(target.shape).to(target.device)
        target_out = (target > rand).long()
    else:
        target[torch.bitwise_and(target<0.05,target>=0)]=0
        target[target>0.65]=1
        target[torch.bitwise_and(target!=0,target!=1)]=-1
        target_out = target
    # print(target[:5],rand[:5],target_out[:5])
    target_out[:,-1]=target_in[:,-1]
    target_out[target_in==-1]=-1
    return target_out

    
def main(args):

    with wandb.init(project="quality", config=args):
        config = wandb.config
        d = datetime.datetime.now()
        wandb.run.name = f"{str(d.year)[-2:]}-{d.month}-{d.day}:{d.hour}.{str(d.minute)[0]}"
        wandb.entity='dereyly'
        wandb.project='qual-attr'
        # config=args
        model = models.AttrModelOld(
            encoder=config.encoder,
            pretrained=config.pretrained,
            num_classes=[2]*num_attr, #[2,2,3,1,1], ,
            drop_rate=config.dropout,
            # num_feat=args.num_feat
        ).cuda()
        if config.weights:
            print('load model....', config.weights)
            st = torch.load(config.weights)
            model.load_state_dict(st,strict=False)
        if not config.weights and config.end_epoch==0:
            raise Exception("no weights in test")

        if len(config.resume_path) > 0:
            st_dict = torch.load(config.resume_path)
            model.load_state_dict(st_dict)
            print(config.resume_path)
        train(model, config)


def train(model, config):
    limit=-1
    os.makedirs(config.snapshots, exist_ok=True)

    optimizer = getattr(optim, config.optimizer)(
        model.parameters(), lr=config.lr
    )

    scaler = amp.GradScaler()

    train_loaders, test_loader = data_loaders(
        config
    )
    criterion = getattr(losses, config.loss)()
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.step, eta_min=args.min_lr)
    # wandb.watch(model, criterion, log="all", log_freq=200)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loaders[0]),
        eta_min=1e-6,
        T_mult = 2
    )
    snapshots = Path(config.snapshots)
    best_acc=0
    for epoch in range(config.start_epoch, config.end_epoch):

        train_epoch(
            train_loaders,
            model,
            scaler,
            criterion,
            optimizer,
            scheduler,
            epoch,
            config,
            limit=limit
        )
        step = (epoch+1)*len(train_loaders[0])
        acc =  validation(test_loader, model, criterion, config,step)
        if torch.isnan(acc):
            break
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

      
        if acc > best_acc:  # epoch % config.step ==1:
            torch.save(
                model.state_dict(),
                f"{config.snapshots}/bst_{wandb.run.name}_{config.encoder}_{config.data_name}.pth"

            )
            best_acc = acc
        torch.save(
            model.state_dict(),
            f"{config.snapshots}/last_{wandb.run.name}_{config.encoder}_{config.data_name}.pth"
        )
    if  config.end_epoch==0:
        acc =  validation(test_loader, model, criterion, config, 0)
        
        #ToDO trace
    wandb.finish()

def train_epoch(
    data_loaders,
    model,
    scaler,
    criterion,
    optimizer,
    scheduler,
    epoch,
    config,
    limit=-1
):
    model.train()
    clipping_value = 5  # arbitrary value of your choosing
    torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
    loss_handler = AverageMeter()
    criterion_absolute_loss = nn.MSELoss().cuda()
    tq = tqdm(total=len(data_loaders[0]) * config.batch_size)
    print('-----len ---',len(data_loaders[0]))
    for i, (image,target,_) in enumerate(data_loaders[0]):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # target = unroll_prob(target,epoch)
        with amp.autocast():
            output = model(image)
            
            loss = 0
            for cls in range(target.shape[1]):
                t=target[:,cls].clone().long()
                loss+=criterion(output[cls][t!=-1], t[t!=-1])
            # if i>10:
            #     break
           

        scaler.scale(loss).backward()

        # if (i + 1) % config.step == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_handler.update(loss)

        tq.set_description(
            "Epoch {}, lr {:.2e}".format(epoch + 1, get_learning_rate(optimizer))
        )

        tq.set_postfix(
            loss="{:.4f}".format(loss_handler.avg),
        )
        tq.update(config.batch_size)
        scheduler.step()
        wandb.log(
            {
            'train/loss':loss_handler.avg,
            'train/lr':float(get_learning_rate(optimizer))
            },
            step=i+len(data_loaders[0])*epoch)
        if limit!=-1 and i*image.shape[0]>limit:
            break
    tq.close()


def validation(data_loader, model, criterion, config, step):
    #self.attr_names
    model.eval()
    attr_dict={'GaussNoise':0,'MotionBlur':1,'Downscale':2,'ISONoise':3,'JpegCompression':4,'all':5} #,'all':6}
    if dbg:
        for name,v in attr_dict.items():
            os.makedirs(f'{dbg_path}/{name}',exist_ok=True)
    attr_names =  {v: k for k, v in attr_dict.items()}
    criterion_absolute_loss = nn.MSELoss().cuda()
    loss_handler = AverageMeter()
    loss_handelrs = []
    for k in range(num_attr):
        loss_handelrs.append(AverageMeter())
     
    count = 0
    user_stats_phone=defaultdict(int)
    user_stats_mirror=defaultdict(int)
    user_stats_all=defaultdict(int)
    err_paths={}
    with torch.no_grad():
        for (image, target, path) in tqdm(data_loader):
            count += image.shape[1]
            # if count >= limit:
            #     break
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            loss = 0
            for cls in range(target.shape[1]):
                # t=target[:,cls].clone().long()
                t=target[:,cls]
                # print(cls,(t!=-1).sum(),t[:10])
                out= output[cls].argmax(dim=1)
                loss_loc=(out[t!=-1]==t[t!=-1]).float().mean()
                if not torch.isnan(loss_loc):
                    # print(cls,(t!=-1).sum(),loss_loc)
                    loss_handelrs[cls].update(loss_loc)
                    loss+=loss_loc
                
                

            

            loss_handler.update(loss/num_attr)
            
    log_test={}
    log_test["val/mean_acc"] = loss_handler.avg
    res_str=''
    for k in range(num_attr):
        res_str+=f'{loss_handelrs[k].avg:3f} '
        # lss_avg.append(loss_handelrs[k].avg)
        log_test[f"val/{attr_names[k]}"] = loss_handelrs[k].avg
    print('***************************************',res_str)
    wandb.log(log_test, step=step)  # ,commit=false)
    return loss_handler.avg


def data_loaders(config,fold=0):
    

    train_dataset = DatasetDir(f'{config.data_dir}')
    test_dataset = DatasetDir(f'/media/dereyly/ssd4/ImageDB/art/crops/portrait_hd_full', state='test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
   
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return [train_loader], test_loader


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--encoder", type=str, default="tf_efficientnet_b3_ns")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--loss", type=str, default="focal")
    parser.add_argument("--size", type=int, default=256)

    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=0)

    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-3)



    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/dereyly/ssd4/ImageDB/art/crops/all_paint_full",
    )
    parser.add_argument(
        "--data_real",
        type=str,
        default="/media/dereyly/ssd4/ImageDB/xix/real_2.6/images",
    )

    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--snapshots", type=str, default="//media/dereyly/data_hdd/models/qual-attr")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--resume_path", type=str, default="")
    # parser.add_argument("--hard_aug", action="store_true")
    parser.add_argument("--num-feat", type=int, default=1280)
    # parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data_name", type=str, default='aug')
    args = parser.parse_args()

    main(args)
