import torch
from torch.nn import functional as F
from dinoseg import DinoSegmenter,load_dino_weights

def main():
    torch.set_num_threads(4)
    m=DinoSegmenter('outputs/round6_dino_model',config_only=True,head_variant='global_local').cuda().eval()
    state=torch.load('outputs/runs/round17_control640/best.pth',map_location='cpu',weights_only=False)
    load_dino_weights(m,state,allow_head_upgrade=True)
    x=torch.rand(1,3,1024,1024,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=m._single(F.interpolate(x,size=(640,640),mode='bilinear',align_corners=False,antialias=True))
        expected=F.interpolate(expected,size=(256,256),mode='bilinear',align_corners=False)
        actual=m(x)
    assert torch.equal(expected,actual),'Zero initialized fusion must preserve global logits'
    assert actual.shape==(1,9,256,256) and torch.isfinite(actual).all()
    bad={'model':dict(state['model']),'config':state['config']};bad['model'].pop('classifier.2.weight')
    try:load_dino_weights(m,bad,allow_head_upgrade=True)
    except RuntimeError:pass
    else:raise AssertionError('Missing original weights accepted')
    print('GLOBAL_LOCAL_TESTS_OK',flush=True)

if __name__=='__main__':main()
