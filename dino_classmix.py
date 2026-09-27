"""Offline DACS-inspired ClassMix; true source labels override target pseudo labels."""
import random
import torch
from torch.nn import functional as F

def classmix(source, truth, target, pseudo, strong=True):
    if source.shape != target.shape or truth.shape != pseudo.shape:
        raise ValueError('ClassMix requires paired equal-size batches')
    masks=[]
    for labels in truth:
        classes=torch.unique(labels);classes=classes[classes!=0]
        chosen=classes[torch.randperm(len(classes),device=classes.device)[:max(1,(len(classes)+1)//2)]]
        # Every available true barren region is transferred, never invented from confidence.
        chosen=torch.unique(torch.cat([chosen,classes[classes==5]]))
        masks.append(torch.isin(labels,chosen))
    mask=torch.stack(masks)
    image=torch.where(mask[:,None],source,target)
    labels=torch.where(mask,truth,pseudo)
    if strong:
        contrast=random.uniform(.7,1.3);brightness=random.uniform(.7,1.3)
        mean=image.mean((2,3),keepdim=True)
        image=((image-mean)*contrast+mean)*brightness
        image=image.clamp(0,1)
        if random.random()<.5:
            image=F.avg_pool2d(F.pad(image,(1,1,1,1),mode='reflect'),3,1)
    return image,labels,mask
