"""Test confidence/agreement filtering and pseudo-label loss ignore behaviour."""
import torch
import numpy as np
from dino_pseudo import filter_labels,class_thresholds,apply_thresholds
from improvedseg import segmentation_loss

def main():
    probs=torch.zeros(1,9,1,4)
    probs[0,1,0,0]=.97;probs[0,2,0,0]=.03
    probs[0,1,0,1]=.90;probs[0,2,0,1]=.10
    probs[0,1,0,2]=.98;probs[0,2,0,2]=.02
    probs[0,0,0,3]=.99;probs[0,1,0,3]=.01
    votes=torch.tensor([[[[1,1,1,0]]],[[[1,1,1,0]]],[[[1,1,2,0]]],[[[2,1,2,0]]]])
    labels,keep=filter_labels(probs,votes)
    assert labels.tolist()==[[[1,0,0,0]]]
    assert keep.tolist()==[[[True,False,False,False]]]
    logits=torch.randn(1,9,1,4,requires_grad=True)
    loss=segmentation_loss(logits,labels,0.);loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[:,:,:,1:].abs().sum()==0
    zero=segmentation_loss(logits,torch.zeros_like(labels),0.)
    assert zero.item()==0 and torch.isfinite(zero)
    hist=np.zeros((9,1001),dtype=np.int64)
    hist[1,920]=40;hist[1,980]=60
    hist[2,900]=90;hist[2,960]=10;hist[3,1000]=100
    thresholds=class_thresholds(hist)
    np.testing.assert_allclose(thresholds[:5],[1,.98,.92,.99,.99])
    ids,accepted=apply_thresholds(np.array([1,2,0,3],dtype=np.uint8),
        np.floor(np.array([.96,.96,1.,1.])*65535).astype(np.uint16),np.array([True,True,True,False]),thresholds)
    assert ids.tolist()==[0,2,0,0] and accepted.tolist()==[False,True,False,False]
    print('PSEUDO_TEST_OK: fixed/adaptive thresholds, clipping, agreement, class zero and ignored gradients',flush=True)

if __name__=='__main__':main()

