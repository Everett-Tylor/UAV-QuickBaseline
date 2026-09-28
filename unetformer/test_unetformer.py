import torch
from unetformer_pipeline import make_model,forward,loss_fn,metrics

def main():
    torch.set_num_threads(4)
    model=make_model('work/unetformer_pretrained/resnet18_swsl.pth').cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    weights=torch.ones(9,device='cuda');weights[0]=0
    for size,batch in [(512,4),(768,2)]:
        x=torch.rand(batch,3,size,size,device='cuda')
        y=torch.randint(0,9,(batch,size,size),device='cuda')
        model.train()
        with torch.autocast('cuda',dtype=torch.float16):
            z=forward(model,x);assert isinstance(z,tuple)
            loss=loss_fn(z,y,weights)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            pred=forward(model,x)
        assert pred.shape==(batch,9,size,size) and torch.isfinite(pred).all()
        print('GPU_TRAIN_EVAL_OK',size,batch,float(loss.detach()),torch.cuda.max_memory_allocated()/1e9,flush=True)
    z=torch.randn(1,9,16,16,device='cuda',requires_grad=True)
    y=torch.zeros(1,32,32,device='cuda',dtype=torch.long)
    loss=loss_fn(z,y,weights);assert loss.item()==0;loss.backward()
    h=torch.eye(9,dtype=torch.int64);h[0,0]=0;assert metrics(h)['mIoU']==1.
    print('CHECKS_OK',flush=True)

if __name__=='__main__':main()
