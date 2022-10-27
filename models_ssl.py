import torch
import torch.nn as nn
import timm

# from timm.models.resnet import ResNet
import torch.nn.functional as F
from copy import deepcopy
# from lightly.data import LightlyDataset
# from lightly.data import SimCLRCollateFunction
# from lightly.loss import NTXentLoss
# from lightly.models.modules import SimCLRProjectionHead
import vissl
from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights


class SimCLR(torch.nn.Module):
    def __init__(self, backbone, feature_dim=256, out_dim=128):
        super().__init__()
        self.backbone = backbone
        # self.projection_head = SimCLRProjectionHead(feature_dim, feature_dim, out_dim)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        # z = self.projection_head(h)
        return h



class AttrLightly(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes=[1,1,3,1,1],
        drop_rate=0.0,
        weights = None,
        # drop_path_rate=0.0,
        num_feat = 1536
    ):
        super().__init__()
        pretrained = True if weights is None else False
        net = timm.create_model(
            encoder,
            num_classes=0,
            drop_rate=drop_rate,
            # features_only=True,
        )
        backbone = nn.Sequential(
            *list(net.children())[:-1],
        )
        self.model=SimCLR(backbone)
        st = torch.load(weights)
        self.model.load_state_dict(st,strict=False)

        self.model.eval()
        # num_feat = self.model.feature_info.info[-1]["num_chs"]
        #num_feat = 1536 #2048 #1536
        attrs=[]
        # num_feat = self.model.get_classifier().in_features
        # num_feat = list(self.model.children())[-2].in_features
        for cls in num_classes:
            attrs.append(nn.Linear(num_feat,cls))
        self.attrs = nn.ModuleList(attrs)
        

    def forward(self, x):
        # x = self.bn(x)
        with torch.no_grad():
            feat = self.model(x) #[-1]


        res=[]
        for clsassif in self.attrs:
            res.append(clsassif(feat))
        
        return torch.stack(res)





class AttrVissl(nn.Module):
    def __init__(
        self,
        encoder=None,
        num_classes=[1,1,3,1,1],
        weights = None,
        # drop_path_rate=0.0,
        num_feat = 7392
    ):
        super().__init__()
        #'/media/remote/media/dereyly/data_hdd/progs/ssl/vissl/configs/config/benchmark/linear_image_classification/imagenet1k/models/regnet128Gf.yaml'
        cfg = [
          # 'config=fairness/hateful_memes/models/regnet32gf.yaml',
          # 'config=benchmark/linear_image_classification/imagenet1k/models/regnet32Gf.yaml',
          # 'config=/benchmark/linear_image_classification/inaturalist18/models/regnet32Gf.yaml',
          # 'config=benchmark/linear_image_classification/imagenet1k/models/regnet32Gf.yaml',
          # 'config=feature_extraction/trunk_only/regnet128Gf_res5.yaml'
          'config=benchmark/linear_image_classification/openimages/models/regnet128Gf.yaml',
          # 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/media/dereyly/data_hdd/models/ssl/vissl/seer_regnet32gf_model_iteration244000.torch', 
          'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/media/dereyly/data_hdd/models/ssl/vissl/model_final_checkpoint_phase0.torch',
          'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
          'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
          'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
          'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
          # 'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["flatten", ["Identity", []]]]' # Extract only the res5avg features.
          # 'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=["flatten"]' # Extract only the res5avg features.
        ]
        # Compose the hydra configuration.
        cfg = compose_hydra_configuration(cfg)
        # Convert to AttrDict. This method will also infer certain config options
        # and validate the config is valid.
        _, cfg = convert_to_attrdict(cfg)



        self.model = vissl.models.build_model(cfg.MODEL, cfg.OPTIMIZER)
        weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
        # Initializei the model with the simclr model weights.
        init_model_from_consolidated_weights(
            config=cfg,
            model=self.model,
            state_dict=weights,
            state_dict_key_name="classy_state_dict",
            skip_layers=[],  # Use this if you do not want to load all layers
        )
        # self.model.eval()
        # num_feat = self.model.feature_info.info[-1]["num_chs"]
        #num_feat = 1536 #2048 #1536
        attrs=[]
        # num_feat = self.model.get_classifier().in_features
        # num_feat = list(self.model.children())[-2].in_features
        for cls in num_classes:
            attrs.append(nn.Linear(num_feat,cls))
        self.attrs = nn.ModuleList(attrs)
        

    def forward(self, x):
        # x = self.bn(x)
        with torch.no_grad():
            feats = self.model(x) #[-1]
        for feat in feats:
            print(feat.shape)
        feat = feats[0]
        res=[]
        for clsassif in self.attrs:
            res.append(clsassif(feat))
        
        out = torch.stack(res)
        # print(out.shape)
        return out


class AttrVisslMLP(nn.Module):
    def __init__(
        self,
        encoder=None,
        num_classes=[1,1,3,1,1],
        weights = None,
        # drop_path_rate=0.0,
        num_feat = 26912,
        num_feat2=4096
    ):
        super().__init__()
        #'/media/remote/media/dereyly/data_hdd/progs/ssl/vissl/configs/config/benchmark/linear_image_classification/imagenet1k/models/regnet128Gf.yaml'
        cfg = [
          # 'config=fairness/hateful_memes/models/regnet32gf.yaml',
          # 'config=benchmark/linear_image_classification/imagenet1k/models/regnet32Gf.yaml',
          'config=/benchmark/linear_image_classification/inaturalist18/models/regnet32Gf.yaml',
          # 'config=benchmark/linear_image_classification/imagenet1k/models/regnet32Gf.yaml',
          # 'config=feature_extraction/trunk_only/regnet128Gf_res5.yaml'
          # 'config=benchmark/linear_image_classification/openimages/models/regnet128Gf.yaml',
          'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/media/dereyly/data_hdd/models/ssl/vissl/seer_regnet32gf_model_iteration244000.torch', 
          # 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/media/dereyly/data_hdd/models/ssl/vissl/model_final_checkpoint_phase0.torch',
          'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
          'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
          'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
          'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
          # 'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["flatten", ["Identity", []]]]' # Extract only the res5avg features.
          # 'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=["flatten"]' # Extract only the res5avg features.
        ]
        # Compose the hydra configuration.
        cfg = compose_hydra_configuration(cfg)
        # Convert to AttrDict. This method will also infer certain config options
        # and validate the config is valid.
        _, cfg = convert_to_attrdict(cfg)


        # ToDO add GN fp16
        # add SilU
        self.model = vissl.models.build_model(cfg.MODEL, cfg.OPTIMIZER)
        weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
        # Initializei the model with the simclr model weights.
        init_model_from_consolidated_weights(
            config=cfg,
            model=self.model,
            state_dict=weights,
            state_dict_key_name="classy_state_dict",
            skip_layers=[],  # Use this if you do not want to load all layers
        )
        # self.model.eval()
        # num_feat = self.model.feature_info.info[-1]["num_chs"]
        #num_feat = 1536 #2048 #1536
        attrs=[]
        # num_feat = self.model.get_classifier().in_features
        # num_feat = list(self.model.children())[-2].in_features
        # self.bn= nn.BatchNorm1d(num_feat)
        self.mlp=nn.Linear(num_feat,num_feat2)

        self.activation=torch.nn.SiLU(inplace=True)
        for cls in num_classes:
            attrs.append(nn.Linear(num_feat2,cls))
        self.attrs = nn.ModuleList(attrs)
        

    def forward(self, x):
        # x = self.bn(x)
        with torch.no_grad():
            feats = self.model(x) #[-1]
        feats_flat=[]
        for feat in feats:
            # print(feat.shape)
            feats_flat.append(torch.flatten(feat, start_dim=1))
        feat = torch.cat(feats_flat,dim=1)
        # print(feat.shape)
        # feat=self.activation(feat)
        # feat = self.bn(feat)
        feat = self.activation(feat)
        y=self.mlp(feat)
        res=[]
        for clsassif in self.attrs:
            res.append(clsassif(y))
        
        out = torch.stack(res)
        # print(out.shape)
        return out


class AttrVisslHead(nn.Module):
    def __init__(
        self,
        encoder=None,
        num_classes=[1,1,3,1,1],
        weights = None,
        # drop_path_rate=0.0,
        num_feat = 11352, #5800,
        num_feat2= 4096
    ):
        super().__init__()
        #'/media/remote/media/dereyly/data_hdd/progs/ssl/vissl/configs/config/benchmark/linear_image_classification/imagenet1k/models/regnet128Gf.yaml'
        # 'config=benchmark/linear_image_classification/imagenet1k/models/regnet32Gf.yaml',
        # 'config=feature_extraction/trunk_only/regnet128Gf_res5.yaml'
        # 'config=fairness/hateful_memes/models/regnet32gf.yaml',
        # 'config=benchmark/linear_image_classification/imagenet1k/models/regnet32Gf.yaml',
        cfg = [
        
          # 'config=/benchmark/linear_image_classification/inaturalist18/models/regnet32Gf.yaml',
          
          'config=benchmark/linear_image_classification/openimages/models/regnet128Gf.yaml',
          # 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/media/dereyly/data_hdd/models/ssl/vissl/seer_regnet32gf_model_iteration244000.torch', 
          'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/media/dereyly/data_hdd/models/ssl/vissl/model_final_checkpoint_phase0.torch',
          'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
          'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
          'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
          'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
          'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[ ["res3", ["Identity", []]],["res4", ["Identity", []]],["avgpool", ["Identity", []]] ]', 
          # 'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=["flatten"]' # Extract only the res5avg features.
        ]
        # Compose the hydra configuration.
        cfg = compose_hydra_configuration(cfg)
        # Convert to AttrDict. This method will also infer certain config options
        # and validate the config is valid.
        _, cfg = convert_to_attrdict(cfg)


        # ToDO add GN fp16
        # add SilU
        self.model = vissl.models.build_model(cfg.MODEL, cfg.OPTIMIZER)
        weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
        # Initializei the model with the simclr model weights.
        init_model_from_consolidated_weights(
            config=cfg,
            model=self.model,
            state_dict=weights,
            state_dict_key_name="classy_state_dict",
            skip_layers=[],  # Use this if you do not want to load all layers
        )
        # self.model.eval()
        # num_feat = self.model.feature_info.info[-1]["num_chs"]
        #num_feat = 1536 #2048 #1536
        attrs=[]
        num_feat_conv3 = 1056 #696
        num_feat_conv4 = 2904 #1392
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_feat_conv3, num_feat_conv3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_conv3),
            nn.SiLU(),
            nn.Conv2d(num_feat_conv3, num_feat_conv3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_conv3),
            nn.SiLU(),
            nn.Conv2d(num_feat_conv3, num_feat_conv3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_conv3),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_feat_conv4, num_feat_conv4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_feat_conv4),
            nn.SiLU(),
            nn.Conv2d(num_feat_conv4, num_feat_conv4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_feat_conv4),
            nn.SiLU(),
            
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp=nn.Linear(num_feat,num_feat2)

        self.activation=torch.nn.SiLU(inplace=True)
        for cls in num_classes:
            attrs.append(nn.Linear(num_feat2,cls))
        self.attrs = nn.ModuleList(attrs)
        

    def forward(self, x):
        # x = self.bn(x)
        with torch.no_grad():
            feats = self.model(x) #[-1]
        feats_flat=[]
        # for feat in feats:
        #     print(feat.shape)
        #     feats_flat.append(torch.flatten(feat, start_dim=1))
        x = self.conv3(feats[0])
        x=torch.flatten(self.avgpool(x), start_dim=1)
        y = self.conv4(feats[1])
        y=torch.flatten(self.avgpool(y), start_dim=1)
        z=torch.flatten(self.avgpool(feats[2]), start_dim=1)
        # print(x.shape,y.shape,z.shape)
        feat = torch.cat([x,y,z],dim=1)
        # print(feat.shape)
        # feat=self.activation(feat)
        # feat = self.bn(feat)
        feat = self.activation(feat)
        y=self.mlp(feat)
        res=[]
        for clsassif in self.attrs:
            res.append(clsassif(y))
        
        out = torch.stack(res)
        # print(out.shape)
        return out