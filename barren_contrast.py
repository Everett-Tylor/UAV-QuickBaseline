"""Regional prototype contrast inspired by ReCo, not an exact reproduction."""
import torch
from torch.nn import functional as F

def regional_contrast(features,target,logits,temperature=.2,max_queries=64):
    # Pure feature cells avoid assigning mixed boundaries or ignored pixels to classes.
    labels=F.one_hot(target,9).permute(0,3,1,2).float()
    occupancy=F.adaptive_avg_pool2d(labels,features.shape[-2:])
    purity,classes=occupancy.max(1);valid=(purity>=.9)&(classes!=0)
    vectors=F.normalize(features.float().permute(0,2,3,1).reshape(-1,features.shape[1]),dim=1)
    classes=classes.flatten();valid=valid.flatten()
    probs=F.interpolate(logits.detach().float(),size=features.shape[-2:],mode='bilinear',align_corners=False).softmax(1).permute(0,2,3,1).reshape(-1,9)
    ids=[];prototypes=[];positions={}
    for c in [1,5,6,7]:
        index=torch.where(valid&(classes==c))[0]
        if len(index)<8:continue
        ids.append(c);positions[c]=index
        prototypes.append(F.normalize(vectors[index].detach().mean(0),dim=0))
    if 5 not in ids or len(ids)<2:return features.float().sum()*0
    centers=torch.stack(prototypes);losses=[]
    for j,c in enumerate(ids):
        index=positions[c]
        # Hard examples by true-class uncertainty; cap per class avoids pixel imbalance.
        chosen=index[torch.topk(1-probs[index,c],min(max_queries,len(index))).indices]
        scores=vectors[chosen]@centers.T/temperature
        losses.append(F.cross_entropy(scores,torch.full((len(chosen),),j,device=scores.device,dtype=torch.long)))
    return torch.stack(losses).mean()
