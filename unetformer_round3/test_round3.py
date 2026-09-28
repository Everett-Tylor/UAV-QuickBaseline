import torch
from PIL import Image
from round3 import make_model,forward,hard_pixel_loss,freeze_bn,infer,MODES

def main():
    torch.set_num_threads(4)
    model=make_model().cuda()
    model.load_state_dict(torch.load('outputs/unetformer_tuned/run/best.pth',map_location='cpu',weights_only=True)['model'])
    model.train();freeze_bn(model)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6);scaler=torch.amp.GradScaler('cuda')
    w=torch.ones(9,device='cuda');w[0]=0
    for size in [448,512,576]:
        x=torch.rand(2,3,size,size,device='cuda');y=torch.randint(0,9,(2,size,size),device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):loss=hard_pixel_loss(forward(model,x),y,w)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    z=torch.randn(1,9,16,16,device='cuda',requires_grad=True)
    y=torch.ones(1,16,16,device='cuda',dtype=torch.long);y[:,:,:8]=0
    hard_pixel_loss(z,y,w).backward();assert (z.grad[:,:,:,:8]==0).all()
    z.grad=None;y.zero_();l=hard_pixel_loss(z,y,w);assert l.item()==0;l.backward()
    model.eval()
    with Image.open('work/test2_images/test2_1.png') as im:
        for mode in MODES.values():
            p=infer(model,im.convert('RGB'),mode);assert p.shape==(1024,1024) and p.max()<=8
    print('ROUND3_GPU_LOSS_IGNORE_TTA_OK',flush=True)

if __name__=='__main__':main()
