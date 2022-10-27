export CUDA_VISIBLE_DEVICES=3


# python train_attr_ssl_pretr.py --batch-size=16 --end-epoch=7  --loss=rfocal  --data_name=tsv_all  --num-feat=3712
# python train_attr_ssl_pretr.py --size=224 --batch-size=4 --end-epoch=1  --pretrained --loss=rfocal  --data_name=tsv_all  --num-feat=7392

python train_attr.py --batch-size=64 --end-epoch=33  --pretrained  --num-feat=1568  --encoder=tf_efficientnet_b3_ns --loss=focal 
# --data_name=pseudo_vissl2_soft_plus  --size=256 
# --weights=/media/dereyly/data_hdd/models/sppof-attr3/last_22-6-22:0.3_tf_efficientnet_b3_ns_pseudo_vissl2_soft_plus.pth
