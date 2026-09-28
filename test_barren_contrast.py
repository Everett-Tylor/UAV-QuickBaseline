import torch
from barren_contrast import regional_contrast

def main():
    torch.manual_seed(8)
    y=torch.ones(1,8,8,dtype=torch.long);y[:,:,4:]=5;y[:,:2]=0
    f=torch.randn(1,4,8,8,requires_grad=True);z=torch.randn(1,9,8,8)
    loss=regional_contrast(f,y,z);assert torch.isfinite(loss) and loss>0
    loss.backward();assert torch.isfinite(f.grad).all() and f.grad[:,:,:2].abs().sum()==0
    assert regional_contrast(f,torch.zeros_like(y),z)==0
    assert regional_contrast(f,torch.ones_like(y),z)==0
    separated=torch.zeros_like(f);separated[:,0,:,:4]=1;separated[:,1,:,4:]=1
    assert regional_contrast(separated,y,z)<loss
    print('CONTRAST_TESTS_OK')

if __name__=='__main__':main()
