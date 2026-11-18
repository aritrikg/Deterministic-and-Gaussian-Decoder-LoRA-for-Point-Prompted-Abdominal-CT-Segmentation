"""CPU regression tests; fake SAM2 results are NEVER research evidence."""
import copy,json,os,sys,tempfile,time,types,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'base_v4'),str(ROOT/'base_v4/source')]
import torch
torch.set_num_threads(1)
from v5.common import read,folds,programme,scientific_identity,REFERENCE,atomic
from v5.models import RankGate,install_v2,predict_logits,objective
from tests.fake_sam2 import FakeSAM2,FakePromptEncoder,FakeMaskDecoder,FakeImageEncoder

def cfg(variant='det_adaptive_v2'):
    c=read(ROOT/'base_v4/assets/control_config.json');c['v5']={'id':'fixture','variant':variant,'seed':42,'fold':1}
    c['experiment'].update(seed=42,device='cpu');c['runtime']={'encoder_checkpointing':'off'};c['training']['amp']=False
    c['lora'].update(encoder_target_modules=['qkv'],target_modules=['q_proj','out_proj'],encoder_rank=2,decoder_rank=16)
    c['loss'].update(lambda_kl=0,lambda_rank=.01);return c

def fake(c):
    from models.deterministic_sam2 import DeterministicSAM2
    back=FakeSAM2();back.hidden_dim=32;back.image_encoder.conv2=torch.nn.Conv2d(6,32,3,padding=1);back.sam_prompt_encoder=FakePromptEncoder(32,4);back.sam_mask_decoder=FakeMaskDecoder(32)
    m=DeterministicSAM2(c,'cpu',backbone=back);m._run_config=c
    if c['v5']['variant'].endswith('adaptive_v2'):install_v2(m)
    # Nonzero B is needed to observe task-gate derivatives after zero-B initialization.
    for n,p in m.named_parameters():
        if n.endswith('lora_B'):torch.nn.init.normal_(p,std=.02)
    return m

class Integrity(unittest.TestCase):
    def test_historical_source_unchanged(self):
        from protocol import verify_source
        self.assertEqual(verify_source(),'7a1c30bb594633075838cfb886cd77d494dd7e605ed6ab076905a690eb2584cb')
    def test_folds_and_ledger(self):
        f=folds();self.assertEqual(len(set(c for s in f.values() for c in s['test'])),30)
        self.assertEqual(f[1]['validation'],['img0006','img0007','img0023','img0027','img0040'])
        from run_budget import merge,ORIGINAL_ACCOUNTED_HOURS
        paths=list((ROOT/'evidence').glob('full_run_ledger_*.json'));s=merge(paths+paths)
        self.assertAlmostEqual(ORIGINAL_ACCOUNTED_HOURS+sum(x['elapsed_seconds'] for x in s.values())/3600,35.726407254264196)
    def test_unclosed_accounting_rejected(self):
        from run_budget import merge
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';x=read(next((ROOT/'evidence').glob('full_run_ledger_*.json')))
            next(iter(x['sessions'].values()))['closed']=False;atomic(p,x)
            with self.assertRaises(Exception):merge([p])
    def test_gate_hard_forward_task_gradient(self):
        gate=RankGate(2);a=torch.tensor([.2,1.]);mix,p=gate.choose(a,True)
        self.assertTrue(torch.equal(mix.detach(),torch.nn.functional.one_hot(p.argmax(-1),3).float()))
        (mix*torch.tensor([[1.,2.,5.],[3.,7.,1.]])).sum().backward()
        self.assertGreater(gate.logits.grad.abs().sum().item(),0)
    def test_v1_hard_gate_has_no_task_gradient(self):
        from models.rank_controller import RankController,RankControllerConfig
        m=RankController(['a'],RankControllerConfig([4,8,16]));rank=m.select_ranks({'a':1.})['a']
        x=torch.ones(2,requires_grad=True);task=(x*rank).sum();g=torch.autograd.grad(task,m.gate,allow_unused=True)[0]
        self.assertIsNone(g);m.sparsity_loss().backward();self.assertIsNotNone(m.gate.grad)
    def test_v2_real_wrapper_gradient_two_organs(self):
        from losses.losses import LossWeights
        c=cfg();m=fake(c);m.train();x=torch.rand(1,1,64,64);y=torch.zeros(1,14,64,64);y[:,1,8:32,8:32]=1;y[:,2,34:55,30:55]=1
        loss,terms=objective(m,x,y,LossWeights(lambda_kl=0,lambda_cal=.1,lambda_rank=0),1,False,3)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(m.v2_gate.logits.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_deterministic_never_fakes_epistemic_samples(self):
        c=cfg('det_v1');m=fake(c);m.eval();e=m.encode_image(torch.rand(1,1,64,64));p=torch.tensor([[[20.,20.]]]);lab=torch.ones(1,1,dtype=torch.int32)
        with self.assertRaises(ValueError):predict_logits(m,e,p,lab,k=4)
    def test_bayesian_mixture_samples_and_gradients(self):
        from models.bayesian_lora import BayesianLoRALinear,BayesianLoRAConfig
        from v5.models import mixture_forward
        # Test the exact production replacement on a Gaussian adapter.
        layer=BayesianLoRALinear(torch.nn.Linear(24,24),BayesianLoRAConfig(in_features=24,out_features=24,rank=16,alpha=16,dropout=0,prior_std=1,init_posterior_std=.05))
        layer.forward=types.MethodType(mixture_forward,layer);gate=RankGate(1);mix,p=gate.choose(torch.tensor([1.]),True);layer._v5_mix=mix[0]
        x=torch.rand(2,24);a=layer(x);b=layer(x);self.assertFalse(torch.equal(a,b));(a.square().mean()+b.square().mean()).backward()
        self.assertIsNotNone(gate.logits.grad);self.assertTrue(torch.isfinite(layer.log_sigma_A.grad).all())
    def test_bayesian_wrapper_v1_fixed_v2_multi_organ_backward(self):
        from models.sam2_wrapper import Sam2BayesianLoRAModel,Sam2BayesianLoRAConfig
        from losses.losses import LossWeights
        class WideEncoder(FakeImageEncoder):
            def __init__(self):super().__init__();self.qkv=torch.nn.Linear(32,32);self.conv0=torch.nn.Conv2d(32,4,3,padding=1);self.conv2=torch.nn.Conv2d(6,32,3,padding=1)
            def forward(self,x):return super().forward(x.repeat(1,11,1,1)[:,:32])
        for variant in ('balora_v1','bayes_fixed_v1','bayes_adaptive_v2'):
            back=FakeSAM2();back.image_encoder=WideEncoder();back.hidden_dim=32;back.sam_prompt_encoder=FakePromptEncoder(32,4);back.sam_mask_decoder=FakeMaskDecoder(32)
            mc=Sam2BayesianLoRAConfig(model_cfg='fixture',checkpoint='',target_modules=['q_proj','out_proj'],rank_bank=[4,8,16],encoder_target_modules=['qkv'],device='cpu')
            m=Sam2BayesianLoRAModel(mc,sam2_model=back);m._run_config=cfg(variant)
            if variant.endswith('adaptive_v2'):install_v2(m)
            if variant=='bayes_fixed_v1':m.rank_controller.requires_grad_(False)
            m.train();x=torch.rand(1,1,64,64);y=torch.zeros(1,14,64,64);y[:,1,8:30,8:30]=1;y[:,2,34:55,30:55]=1
            loss,_=objective(m,x,y,LossWeights(lambda_kl=.0001,lambda_cal=.1,lambda_rank=0),2,False,3);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            if variant.endswith('adaptive_v2'):self.assertGreater(m.v2_gate.logits.grad.abs().sum().item(),0)
    def test_incremental_evaluation_and_coverage(self):
        from v5 import engine
        from utils.runtime_budget import BudgetExpired
        from datasets.btcv_dataset import BTCVSample
        class Tiny:
            case_ids=['fixture']
            def __init__(self,p):self.pairs={'fixture':{'image':p,'label':p}}
            def __len__(self):return 1
            def _read_volume(self,c):
                import numpy as np
                lab=np.zeros((64,64,1),dtype=np.uint8);lab[8:32,8:32]=1;lab[34:55,30:55]=2;return lab,lab
            def __getitem__(self,i):
                x=torch.full((1,1,64,64),.3);y=torch.zeros(1,14,64,64);y[:,1,8:32,8:32]=1;y[:,2,34:55,30:55]=1
                return BTCVSample(x,y,'fixture',[0],torch.ones(1,dtype=torch.bool))
        old_loader,old_check=engine.loaders,engine.check
        try:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'raw';p.write_bytes(b'fixture');engine.loaders=lambda *a:types.SimpleNamespace(dataset=Tiny(p))
                c=cfg('det_v1');m=fake(c);calls=[0]
                def pause(deadline):
                    calls[0]+=1
                    if calls[0]==3:raise BudgetExpired('test target pause')
                engine.check=pause
                with self.assertRaises(BudgetExpired):engine.infer(m,c,'test',Path(d)/'test',time.time()+60)
                self.assertEqual(len(list((Path(d)/'test/records').glob('*.json'))),1)
                engine.check=old_check;r=engine.infer(m,c,'test',Path(d)/'test',time.time()+60)
                self.assertTrue(r['complete']);self.assertEqual(r['records'],2)
                with torch.no_grad():next(p for p in m.parameters() if p.requires_grad).add_(.1)
                with self.assertRaises(ValueError):engine.infer(m,c,'test',Path(d)/'test',time.time()+60)
        finally:engine.loaders=old_loader;engine.check=old_check
    def test_finite_plan_and_identity(self):
        jobs=programme();self.assertEqual(len(jobs),17);self.assertEqual(len({x['id'] for x in jobs}),17)
        c=cfg();d=copy.deepcopy(c);d['sam2']['checkpoint']='/other/path';d['experiment']['output_dir']='/different';self.assertEqual(scientific_identity(c),scientific_identity(d))
        d['loss']['lambda_cal']+=.1;self.assertNotEqual(scientific_identity(c),scientific_identity(d))
        from v5.common import configuration
        with tempfile.TemporaryDirectory() as out:
            for j in jobs:
                resolved=configuration(j,{'btcv_root':'/fixture'},{'backbone_checkpoint':'/fixture','official_commit':'fixture','backbone_sha256':'fixture'},out,time.time()+30)
                self.assertFalse(resolved['refinement']['enabled']);self.assertEqual(resolved['training']['num_epochs'],20)
                if j['variant']=='det_adaptive_v2':self.assertEqual(resolved['loss']['lambda_rank'],.01)
                if j['variant'].startswith('det'):self.assertEqual(resolved['loss']['lambda_kl'],0)
    def test_volume_dependence(self):
        from v5.report import paired,pair_values
        a=[];b=[]
        for c in range(6):
            for o in range(1,14):
                for z in range(2):
                    a.append({'case_id':str(c),'organ_class':o,'dice':.7+c*.01});b.append({'case_id':str(c),'organ_class':o,'dice':.6+c*.01})
        r=paired(a,b);self.assertEqual(r['independent_units'],6);self.assertAlmostEqual(r['mean_difference'],.1)
        a=[{'case_id':'a','organ_class':1,'dice':1.},{'case_id':'a','organ_class':2,'dice':1.},{'case_id':'b','organ_class':1,'dice':0.}]
        b=[{**r,'dice':0.} for r in a];r=paired(a,b)
        self.assertAlmostEqual(r['mean_difference'],2/3);self.assertAlmostEqual(r['mean_volume_macro_difference'],.5)
    def test_zip_path_escape(self):
        import zipfile
        from run_budget import safe_extract
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','x')
            with self.assertRaises(ValueError):safe_extract(p,Path(d)/'extract')
    def test_point_unet_shape_and_task_gradient(self):
        from v5.models import PointUNet
        m=PointUNet();x=torch.rand(1,1,32,32);point=torch.tensor([[[15.,15.]]]);lab=torch.ones(1,1,dtype=torch.int32)
        y=m.decode_one(x,point,lab);self.assertEqual(tuple(y.shape),(1,1,32,32));y.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_labeled_root_excludes_unlabeled_challenge_test(self):
        from data_root import resolve_btcv_root
        from protocol import split_ids
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'BTCV';train=root/'RawData/Training';(train/'img').mkdir(parents=True);(train/'label').mkdir()
            for c in sum(split_ids().values(),[]):(train/'img'/(c+'.nii.gz')).write_bytes(b'fixture');(train/'label'/('label'+c[3:]+'.nii.gz')).write_bytes(b'fixture')
            test=root/'RawData/Test/img';test.mkdir(parents=True);(test/'img0061.nii.gz').write_bytes(b'unlabeled fixture')
            selected,audit=resolve_btcv_root(root);self.assertEqual(Path(selected),train);self.assertEqual(audit['labeled_cases'],30)
    def test_interrupted_training_matches_uninterrupted(self):
        """Actual adapter updates, optimizer/scheduler/RNG restored mid-epoch."""
        from v5 import engine
        from training import train as training
        from utils.runtime_budget import BudgetExpired
        from datasets.btcv_dataset import collate_btcv,BTCVSample
        from torch.utils.data import DataLoader,Dataset
        class Tiny(Dataset):
            total_clips_before_filter=3
            def __init__(self,path):self.case_ids=['fixture'];self.pairs={'fixture':{'image':path,'label':path}}
            def __len__(self):return 3
            def __getitem__(self,i):
                x=torch.full((1,1,64,64),.2+i*.1);y=torch.zeros(1,14,64,64);y[:,1,10:35,10:35]=1
                return BTCVSample(x,y,'fixture',[i],torch.ones(1,dtype=torch.bool))
        old_build,old_loader,old_infer=engine.build,engine.loaders,engine.infer;old_epoch=training.train_one_epoch
        try:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'raw';p.write_bytes(b'fixture')
                engine.build=lambda c,dev:fake(c);engine.loaders=lambda c,r:DataLoader(Tiny(p),batch_size=1,collate_fn=collate_btcv)
                engine.infer=lambda *a,**k:{'complete':True,'mean_dice':.5}
                c=cfg('det_v1');c['loss']['lambda_rank']=0
                engine.fit(c,Path(d)/'whole',time.time()+60)
                def interrupted(*args,**kwargs):
                    cb=kwargs['progress_callback']
                    def pause(progress):cb(progress);raise BudgetExpired('test mid-epoch pause')
                    kwargs['progress_callback']=pause;return old_epoch(*args,**kwargs)
                training.train_one_epoch=interrupted
                with self.assertRaises(BudgetExpired):engine.fit(c,Path(d)/'resume',time.time()+60)
                training.train_one_epoch=old_epoch;engine.fit(c,Path(d)/'resume',time.time()+60)
                a=torch.load(Path(d)/'whole/research/last.pt',weights_only=False,map_location='cpu');b=torch.load(Path(d)/'resume/research/last.pt',weights_only=False,map_location='cpu')
                self.assertEqual(a['v5_state']['epoch'],8);self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
                self.assertEqual(a['v5_state']['history'][-1]['updates'],3)
                for n,v in a['model_state_dict'].items():self.assertTrue(torch.equal(v,b['model_state_dict'][n]),n)
        finally:engine.build=old_build;engine.loaders=old_loader;engine.infer=old_infer;training.train_one_epoch=old_epoch

if __name__=='__main__':unittest.main(verbosity=2)
