"""Separate v2 controller; original Bayesian/adaptive implementation is untouched.

v2 uses detached predictive-entropy gradient x adapter activation to condition
per-layer categorical preferences. Hard prefix rank in the forward pass,
softmax straight-through derivative in training; this is a surrogate gradient,
not an exact gradient of discrete rank. Same allocator signal in both v2 arms.
"""
import types
import torch
from torch import nn
from models.bayesian_lora import BayesianLoRALinear,DeterministicLoRALinear

def entropy(p):
    q=p.clamp(1e-6,1-1e-6)
    return -(q*q.log()+(1-q)*(1-q).log())

class RankGate(nn.Module):
    def __init__(self,n,temperature=1.):
        super().__init__();self.logits=nn.Parameter(torch.zeros(n,3));self.temperature=temperature
        self.register_buffer('bank',torch.tensor([4.,8.,16.]))
    def probabilities(self,attribution):
        # Maximum layer attribution normalization, centers at 0.5.
        a=attribution.detach().clamp_min(0);a=a/a.max().clamp_min(1e-12)
        bias=4*(a[:,None]-.5)*torch.tensor([-1.,0.,1.],device=a.device)
        return ((self.logits+bias)/self.temperature).softmax(-1)
    def choose(self,attribution,training):
        p=self.probabilities(attribution);hard=torch.nn.functional.one_hot(p.argmax(-1),3).to(p.dtype)
        return hard-p.detach()+p if training else hard,p

def mixture_forward(self,x):
    A,B=self.sample_weights(sample=self.sample_posterior) if isinstance(self,BayesianLoRALinear) else (self.lora_A,self.lora_B)
    dropped=self.dropout(x)
    alpha=self.config.alpha if isinstance(self,BayesianLoRALinear) else self.alpha
    if self._v5_mix is None:
        update=(dropped@A[:16].T@B[:,:16].T)*(alpha/16)
    elif not self.training:
        r=(4,8,16)[int(self._v5_mix.argmax().item())]
        update=(dropped@A[:r].T@B[:,:r].T)*(alpha/r)
    else:
        raw=[dropped@A[:r].T@B[:,:r].T for r in (4,8,16)]
        update=sum(self._v5_mix[i]*raw[i]*(alpha/r) for i,r in enumerate((4,8,16)))
    self.last_adapter_output=update
    return self.base_layer(x)+update

def decoder_modules(model):
    return [(n,m) for n,m in model.backbone.sam_mask_decoder.named_modules() if isinstance(m,(BayesianLoRALinear,DeterministicLoRALinear))]

def install_v2(model):
    modules=decoder_modules(model)
    if not modules:raise ValueError('No decoder adapters')
    if hasattr(model,'rank_controller'):
        # v1 scalar prior has no place in v2; do not retain an unused trainable parameter.
        model.rank_controller=nn.Identity()
    model.v2_gate=RankGate(len(modules)).to(next(model.parameters()).device)
    for _,m in modules:m._v5_mix=None;m.forward=types.MethodType(mixture_forward,m)
    return model

class PointUNet(nn.Module):
    """Custom 4-level 2D U-Net: [windowed CT, positive-point Gaussian] -> mask.
    No organ ID or mask input. Channels 16/32/64/128/256, GroupNorm, bilinear up.
    """
    def __init__(self):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())
        cs=[16,32,64,128,256];self.down=nn.ModuleList([block(2,16)]+[block(cs[i-1],cs[i]) for i in range(1,5)])
        self.up=nn.ModuleList([block(cs[i]+cs[i-1],cs[i-1]) for i in range(4,0,-1)]);self.head=nn.Conv2d(16,1,1)
    def encode_image(self,image):return image
    def decode_one(self,image,points,labels):
        h,w=image.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=image.device),torch.arange(w,device=image.device),indexing='ij')
        xy=points[:,0];g=torch.exp(-((xx[None]-xy[:,0,None,None])**2+(yy[None]-xy[:,1,None,None])**2)/(2*10.**2))
        x=torch.cat([image,g[:,None]],1);skip=[]
        for i,layer in enumerate(self.down):
            if i:x=torch.nn.functional.max_pool2d(x,2)
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x=checkpoint(layer,x,use_reentrant=False)
            else:x=layer(x)
            skip.append(x)
        for i,layer in enumerate(self.up):
            x=torch.nn.functional.interpolate(x,size=skip[-2-i].shape[-2:],mode='bilinear',align_corners=False)
            x=torch.cat([x,skip[-2-i]],1)
            if self.training and torch.is_grad_enabled():x=checkpoint(layer,x,use_reentrant=False)
            else:x=layer(x)
        return self.head(x)

class IndependentSeedEnsemble(nn.Module):
    """Three independently fitted deterministic models, never repeated outputs."""
    def __init__(self,models,cfg):
        super().__init__();self.members=nn.ModuleList(models);self._run_config=cfg
    def encode_image(self,image):return [m.encode_image(image) for m in self.members]

def build(cfg,device,backbone=None):
    v=cfg['v5']['variant'];bayesian=v.startswith('bayes') or v=='balora_v1'
    if v=='point_unet':model=PointUNet().to(device)
    elif bayesian:
        from training.train import build_model
        model=build_model(cfg,device)
    else:
        from models.deterministic_sam2 import DeterministicSAM2
        model=DeterministicSAM2(cfg,device,backbone=backbone)
    if v.endswith('adaptive_v2'):model=install_v2(model)
    elif bayesian and v!='balora_v1':
        # Fixed bank prefix 16, retain Gaussian KL over all stored factors.
        model.rank_controller.requires_grad_(False)
    model._run_config=cfg;return model

def predict_logits(model,embedding,points,labels,k=4,probes=2,forced_rank=None):
    v=model._run_config['v5']['variant'];bayesian=v.startswith('bayes') or v=='balora_v1'
    if v=='det_ensemble':
        if len(model.members)!=3:raise ValueError('Three independent training seeds required')
        return torch.stack([m.decode_one(e,points,labels) for m,e in zip(model.members,embedding)]),{},next(model.parameters()).new_zeros(())
    if not bayesian and k!=1:raise ValueError('Deterministic inference must use one prediction')
    if bayesian and k<2:raise ValueError('Bayesian uncertainty requires at least two predictions')
    ranks={};penalty=next(model.parameters()).new_zeros(())
    if v=='balora_v1' and forced_rank is None:
        ranks,_=model.select_ranks_from_uncertainty(embedding,points,labels,k_samples=probes)
    elif v.endswith('adaptive_v2') and forced_rank is None:
        modules=decoder_modules(model)
        # The probe is detached from final segmentation. Recompute final predictions.
        for _,m in modules:m._v5_mix=None
        with torch.enable_grad():
            if bayesian:probe=model.decode(embedding,points,labels,k_samples=probes,sample=True).sigmoid().mean(0)
            else:probe=model.decode_one(embedding,points,labels).sigmoid()
            acts=[m.last_adapter_output for _,m in modules]
            grads=torch.autograd.grad(entropy(probe).mean(),acts,allow_unused=True,retain_graph=False)
            scores=torch.stack([a.new_zeros(()) if g is None else (a.detach()*g.detach()).abs().mean() for a,g in zip(acts,grads)])
        mix,p=model.v2_gate.choose(scores,model.training);penalty=(p*(model.v2_gate.bank/16)).sum(-1).mean()
        for i,(n,m) in enumerate(modules):m._v5_mix=mix[i];ranks[n]=int(model.v2_gate.bank[p[i].argmax()].item())
    elif hasattr(model,'backbone'):
        ranks={n:int(forced_rank or 16) for n,_ in decoder_modules(model)}
        if v.endswith('adaptive_v2'):
            for _,m in decoder_modules(model):
                m._v5_mix=torch.nn.functional.one_hot(torch.tensor((4,8,16).index(forced_rank or 16),device=points.device),3).float()
    if bayesian:
        if v.endswith('adaptive_v2'):logits=model.decode(embedding,points,labels,k_samples=k,sample=True)
        else:logits=model.decode(embedding,points,labels,k_samples=k,sample=True,ranks=ranks)
    else:
        if forced_rank is not None:
            for _,m in decoder_modules(model):m._active_rank_override=int(forced_rank)
        logits=model.decode_one(embedding,points,labels).unsqueeze(0)
    return logits,ranks,penalty

def objective(model,images,labels,weights,k_train,encoder_amp,kl_count):
    from training.train import _batch_objective,_clear_adapter_graphs
    from losses.losses import compute_total_loss
    from utils.prompting import sample_foreground_points
    v=model._run_config['v5']['variant']
    if v=='balora_v1':return _batch_objective(model,images,labels,weights,k_train,encoder_amp,kl_count)
    embedding=model.encode_image(images)
    if isinstance(embedding,dict):
        from training.train import _promote_embedding_features_to_fp32
        embedding=_promote_embedding_features_to_fp32(embedding)
    segs=[];cals=[];ranks=[];zero=images.new_zeros(())
    for organ in range(1,labels.shape[1]):
        present=labels[:,organ].any((-2,-1))
        if not bool(present.any()):continue
        target=labels[present,organ:organ+1];points=sample_foreground_points(target[:,0]);pl=torch.ones(points.shape[:2],device=images.device,dtype=torch.int32)
        selected=embedding[present]
        logits,_,rank=predict_logits(model,selected,points,pl,k=k_train)
        mean=logits.float().sigmoid().mean(0);loss=compute_total_loss(torch.logit(mean.clamp(1e-6,1-1e-6)),target.float(),zero,zero,weights)
        segs.append(loss.seg);cals.append(loss.calibration);ranks.append(rank)
    seg=torch.stack(segs).mean();cal=torch.stack(cals).mean();rank=torch.stack(ranks).mean()
    kl=model.total_kl()/max(1,kl_count) if (v.startswith('bayes')) else zero
    loss=seg+weights.lambda_cal*cal+weights.lambda_kl*kl+weights.lambda_rank*rank
    return loss,{'seg':seg,'cal':cal,'kl':kl,'rank':rank}
