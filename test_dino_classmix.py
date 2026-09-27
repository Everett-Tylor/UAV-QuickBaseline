import torch
from dino_classmix import classmix

def test():
    torch.manual_seed(42)
    source=torch.ones(2,3,4,4);target=torch.zeros_like(source)
    truth=torch.tensor([[5,5,0,0],[1,1,2,2],[3,3,4,4],[6,6,7,8]]).repeat(2,1,1)
    pseudo=torch.full_like(truth,6);pseudo[:,0,2:]=0
    x,y,mask=classmix(source,truth,target,pseudo,strong=False)
    assert mask[truth==5].all() and not mask[truth==0].any()
    assert torch.equal(y[mask],truth[mask]) and torch.equal(y[~mask],pseudo[~mask])
    assert torch.equal(x[:,0],mask.float())
    assert (y[:,0,2:]==0).all()
    _,empty,m=classmix(source,torch.zeros_like(truth),target,pseudo,strong=False)
    assert not m.any() and torch.equal(empty,pseudo)
    for _ in range(10):
        aug,labels,_=classmix(source,truth,target,pseudo)
        assert torch.isfinite(aug).all() and aug.min()>=0 and aug.max()<=1
        assert labels.dtype==torch.long
    print('CLASSMIX_TESTS_OK')

if __name__=='__main__':test()
