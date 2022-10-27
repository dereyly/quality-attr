import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

# import torch.distributed as distributed
import torch.nn as nn
from tqdm import tqdm

import models
# import models_ssl
from dataset.dataset_dir import DatasetImage
from torch.utils.data import DataLoader
from utils import AverageMeter, MetaData
from collections import defaultdict
import pickle as pkl
import shutil

torch.manual_seed(23)
import random
random.seed(23)
np.random.seed(23)

softm = torch.nn.Softmax(dim=1)
n_attrs = 6

def main(args):
    w_name=args.weights.split('/')[-1][:-4]
    out_path=f"{args.out_dir}/{w_name}"
    os.makedirs(out_path,exist_ok=True)
    is_amp=True

    model = models.AttrModelOld(
        encoder=args.encoder,
        num_classes=[2]*n_attrs, 
        num_feat=args.num_feat
    ).cuda()

    st_dict = torch.load(args.weights)
    model.load_state_dict(st_dict)
    model.eval()
    is_amp=False
   

    
    is_multi=False
    test_dataset = DatasetImage(args.data,is_test=True,is_multi=is_multi)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    result = defaultdict(list)
    count=0
    err = 0
    data_out={}
    with torch.cuda.amp.autocast(enabled=is_amp):
        with torch.no_grad():
            for images, paths, drop in tqdm(test_loader):
                # print(len(images),images[0].shape)
                outs=[]
                if len(images)==0:
                    continue
                for image in images: #ToDo run for one BS
                    image = image.cuda(non_blocking=True)
                    outs += [model(image)]
                # outs += [model(image.flip(2))]
                # outs += [model(image.flip(3))]
               
                op=[]
                
                for k in range(n_attrs):
                    out_loc = torch.zeros(image.shape[0],dtype=np.float,device=outs[0][0].device)
                    for z in range(len(outs)):
                        # if len(outs[z].shape)>2 and outs[z].shape[2]>1:
                        out_loc+=softm(outs[z][k])[:,1]
                        # else:
                        # out_loc+= torch.sigmoid(outs[z][:,k])
                    op.append(out_loc.cpu().numpy()/len(outs))

                op = np.array(op)
                for k in range(len(paths)):
                    key = paths[k].split('/')[-1][:-4] #.split('.')[0]
                    # mop=np.sqrt(np.sum(op[:,k]**2))
                    mop=0.2*np.sqrt(np.sum(op[:,k]**2))+0.25*op[:,k].max()+0.5*op[-1,k]
                    if 1:
                    	shutil.copy(paths[k],f'{out_path}/{mop:.2f}___{op[0,k]:.2f}_{op[1,k]:.2f}_{op[2,k]:.2f}_{op[3,k]:.2f}_{op[4,k]:.2f}_{op[5,k]:.2f}.jpg')
                #     attr=[float(op[n,k]) for n in range(len(op))]
                #     data_out[key]=attr
                    
                #     attr=np.array(attr) #[:-2]
                #     m_attr=0.6*np.sqrt((attr**2).mean())+0.4*attr.max()
                #     err+=m_attr<0.5 #op[0,k]<0.5
                #     count+=1
                # if count % 100000==0:
                #     pkl.dump(data_out, open(f'{args.data}_vissl_bt.pkl','wb'))
            # if count>4:
            #     break
    # print(err/count)
    # pkl.dump(data_out, open(f'{args.data}_vissl_bb.pkl','wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--encoder", type=str, 
            # default="efficientnet_es"
            # default = 'seresnext50_32x4d'
            default = 'tf_efficientnet_b3_ns'
            )
    parser.add_argument("--weights", type=str,
                        
                        default = '/media/dereyly/data_hdd/models/qual-attr/last_22-8-15:11.5_tf_efficientnet_b3_ns_aug.pth'
                        )
    parser.add_argument(
        "--data",
        type=str,
        # default='/media/dereyly/ssd4/ImageDB/art/all'
        default = '/media/dereyly/ssd4/ImageDB/art/fail'
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='/media/dereyly/data_hdd/ImageDB/qual_attr/',
    )
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-feat", type=int, default=1536)
    args = parser.parse_args()

    main(args)
