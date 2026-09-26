"""DINOv3 ViT-B/16 with a multi-layer pyramid decoder for nine-class segmentation."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTModel,DINOv3ViTConfig

def conv(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.GELU())

class MixStyle(nn.Module):
    def __init__(self,p=.5,alpha=.1):
        super().__init__()
        self.p=p
        self.beta=torch.distributions.Beta(alpha,alpha)

    def forward(self,x):
        if not self.training or x.shape[0]<2 or torch.rand(())>=self.p:return x
        value=x.float()
        mean=value.mean((2,3),keepdim=True).detach()
        std=(value.var((2,3),keepdim=True,unbiased=False)+1e-6).sqrt().detach()
        order=torch.randperm(x.shape[0],device=x.device)
        weight=self.beta.sample((x.shape[0],1,1,1)).to(device=x.device)
        new_mean=weight*mean+(1-weight)*mean[order]
        new_std=weight*std+(1-weight)*std[order]
        return (((value-mean)/std)*new_std+new_mean).to(dtype=x.dtype)

class DinoSegmenter(nn.Module):
    def __init__(self,source,config_only=False):
        super().__init__()
        if config_only:
            cfg=DINOv3ViTConfig.from_pretrained(source,local_files_only=True)
            cfg._attn_implementation='sdpa'
            self.backbone=DINOv3ViTModel(cfg)
        else:
            self.backbone=DINOv3ViTModel.from_pretrained(source,local_files_only=True,attn_implementation='sdpa')
        hidden=self.backbone.config.hidden_size
        self.projections=nn.ModuleList([nn.Conv2d(hidden,128,1) for _ in range(4)])
        self.fusions=nn.ModuleList([nn.Sequential(conv(128,128),conv(128,128)) for _ in range(4)])
        self.detail=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU())
        self.classifier=nn.Sequential(conv(160,128),nn.Dropout2d(.1),nn.Conv2d(128,9,1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.frozen_backbone=False
        self.mixstyle=MixStyle()
        self.mixstyle_enabled=False

    def freeze_backbone(self,freeze):
        self.frozen_backbone=freeze
        self.backbone.requires_grad_(not freeze)
        if freeze:self.backbone.eval()

    def train(self,mode=True):
        super().train(mode)
        if self.frozen_backbone:self.backbone.eval()
        return self

    def forward(self,x):
        h,w=x.shape[-2:]
        if h%32 or w%32:raise ValueError('Input dimensions must be divisible by 32')
        normalized=(x-self.mean)/self.std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            states=self.backbone(normalized,output_hidden_states=True).hidden_states
        if len(states)!=13:raise RuntimeError(f'Expected embedding + 12 blocks, got {len(states)}')
        gh,gw=h//16,w//16;prefix=1+self.backbone.config.num_register_tokens
        features=[]
        for state,index,proj,scale in zip([states[i] for i in [3,6,9,12]],[3,6,9,12],self.projections,[4,2,1,.5]):
            state=self.backbone.norm(state)
            tokens=state[:,prefix:]
            if tokens.shape[1]!=gh*gw:raise RuntimeError('Patch/token shape mismatch')
            feature=proj(tokens.transpose(1,2).reshape(x.shape[0],-1,gh,gw))
            if self.mixstyle_enabled and index in (3,6):feature=self.mixstyle(feature)
            features.append(F.interpolate(feature,size=(int(gh*scale),int(gw*scale)),mode='bilinear',align_corners=False))
        fused=None
        for i in [3,2,1,0]:
            value=features[i]
            if fused is not None:value=value+F.interpolate(fused,size=value.shape[-2:],mode='bilinear',align_corners=False)
            fused=value+self.fusions[i](value)
        detail=self.detail(normalized)
        return self.classifier(torch.cat([fused,detail],dim=1))
