import sys,torch
from round4_train import lovasz_softmax
x=torch.randn(2,9,4,4,requires_grad=True);y=torch.randint(1,9,(2,4,4));y[:,0]=0
v=lovasz_softmax(x,y);v.backward();assert torch.isfinite(x.grad).all();assert (x.grad[:,:,0]==0).all()
z=torch.randn(1,9,2,2,requires_grad=True);loss=lovasz_softmax(z,torch.zeros(1,2,2,dtype=torch.long));loss.backward();assert loss.item()==0 and z.grad.abs().sum()==0
labels=torch.ones(1,2,2,dtype=torch.long);good=torch.full((1,9,2,2),-10.);good[:,1]=10.;bad=good.clone();bad[:,1]=-10.;bad[:,2]=10.
assert lovasz_softmax(good,labels)<1e-5 and lovasz_softmax(bad,labels)>.99
print('PASS: finite gradients, ignored pixels, all-ignore target, perfect/wrong predictions')
