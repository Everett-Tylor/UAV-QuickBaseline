"""Verify backward-compatible head upgrade and ignore-aware boundary supervision."""
import torch
from dinoseg import DinoSegmenter,load_dino_weights
from dino_train import boundary_loss

def main():
    torch.set_num_threads(4)
    state=torch.load('outputs/runs/round8_mixstyle640/best.pth',map_location='cpu',weights_only=False)
    old=DinoSegmenter('outputs/round6_dino_model',config_only=True).eval()
    load_dino_weights(old,state)
    new=DinoSegmenter('outputs/round6_dino_model',config_only=True,head_variant='context_boundary').eval()
    load_dino_weights(new,state,allow_head_upgrade=True)
    x=torch.rand(1,3,64,64)
    with torch.no_grad():torch.testing.assert_close(old(x),new(x),rtol=0,atol=0)
    upgraded={'model':new.state_dict(),'config':{'head_variant':'context_boundary'}}
    load_dino_weights(new,upgraded)
    invalid={'model':dict(state['model'])}
    del invalid['model']['mean']
    try:load_dino_weights(new,invalid,allow_head_upgrade=True)
    except RuntimeError:pass
    else:raise AssertionError('Missing original weight was silently accepted')
    y=torch.ones(1,32,32,dtype=torch.long);y[:,:,16:]=2;y[:,:3]=0
    logits=torch.zeros(1,1,32,32,requires_grad=True)
    loss=boundary_loss(logits,y);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[:,:,:4].abs().sum()==0
    assert logits.grad[:,:,8:24,15:17].sum()<0
    assert boundary_loss(logits,torch.zeros_like(y)).item()==0
    print('HEAD_TEST_OK: exact initial outputs, strict upgrade checks, reload, boundary masking',flush=True)

if __name__=='__main__':main()

