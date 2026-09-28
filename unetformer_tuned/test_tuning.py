import copy
import torch
from PIL import Image
from tune import make_model,forward,freeze_bn,update_ema,loss_fn,infer,MODES

def main():
    torch.set_num_threads(4)
    model=make_model().cuda()
    model.load_state_dict(torch.load('outputs/unetformer/run/best.pth',map_location='cpu',weights_only=True)['model'])
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scaler=torch.amp.GradScaler('cuda');model.train();freeze_bn(model)
    bn=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)][0]
    old=bn.running_mean.clone()
    for size in [448,512,576]:
        x=torch.rand(2,3,size,size,device='cuda');y=torch.randint(0,9,(2,size,size),device='cuda')
        w=torch.ones(9,device='cuda');w[0]=0
        with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(forward(model,x),y,w)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        update_ema(ema,model)
    assert torch.equal(old,bn.running_mean)
    model.eval()
    with Image.open('work/test2_images/test2_1.png') as im:
        for name,mode in MODES.items():
            pred=infer(model,im.convert('RGB'),mode)
            assert pred.shape==(1024,1024) and int(pred.max())<=8
    print('TRAIN_BN_EMA_MULTISCALE_CHECKS_OK',flush=True)

if __name__=='__main__':main()
