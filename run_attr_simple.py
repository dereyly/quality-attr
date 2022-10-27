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
# from dataset.dataset_dir import DatasetImage
from collections import defaultdict
import pickle as pkl
import shutil

torch.manual_seed(23)
import random
random.seed(23)
np.random.seed(23)

softm = torch.nn.Softmax(dim=1)
n_attrs = 6
def to_tensor(x):
    # if self.tensor_norm:
    #     x = normalize(x)
    # elif x.dtype == np.uint8:
    x = x / 255
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).float()

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
   

    
    # test_dataset = DatasetImage(args.data,is_test=True,is_multi=is_multi)
    # paths = glob.glob('')
    names = os.listdir(args.data)
    result = defaultdict(list)
    count=0
    err = 0
    data_out={}
    with torch.cuda.amp.autocast(enabled=is_amp):
        with torch.no_grad():
            for name in tqdm(names):
                path=f'{args.data}/{name}'
                image=cv2.imread(path)
                image=to_tensor(image).cuda()

                outs = model(image[None,...])
                print(outs.shape)
                attr=[]
                for k,out in enumerate(outs):
                    outs[k]=softm(out)[:,1]
                    outs[k]=torch.sqrt((outs[k]**2).mean())
                    attr.append(outs[k].cpu().numpy())
                attr=np.array(attr)
                m_attr=0.2*np.sqrt((attr**2).mean())+0.2*attr.max()+attr[-1]
                shutil.copy(path,f'{out_path}/{m_attr:.2f}___{attr[0]:.2f}_{attr[1]:.2f}_{attr[2]:.2f}_{attr[3]:.2f}_{attr[4]:.2f}_{attr[5]:.2f}.jpg')
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
                        
                        default = '/media/dereyly/data_hdd/models/qual-attr/last_22-8-14:13.6_tf_efficientnet_b3_ns_aug.pth'
                        )
    parser.add_argument(
        "--data",
        type=str,
        default='/media/dereyly/ssd4/ImageDB/art/all'
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
