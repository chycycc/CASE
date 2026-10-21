"""Properties of the zero-init residual prefix (CPU, no dataset)."""
import contextlib,io,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
with contextlib.redirect_stdout(io.StringIO()):
 from src.models.common import MultiHeadAttention,DecoderLayer
class GatedPrefixTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(13);torch.set_num_threads(1)
  self.a=MultiHeadAttention(12,12,12,12,2,dropout=0).eval()
  self.x=torch.randn(2,5,12);self.p=nn.Parameter(torch.randn(4,12))
  self.mask=torch.triu(torch.ones(5,5,dtype=torch.bool),1).expand(2,-1,-1)
 def test_zero_gate_recovers_no_prefix_exactly(self):
  base,base_logits=self.a(self.x,self.x,self.x,self.mask)
  for p in [self.p,self.p*1000]:
   out,logits=self.a(self.x,self.x,self.x,self.mask,extra_kv=p,extra_kv_gate=torch.zeros(2))
   torch.testing.assert_close(out,base,rtol=0,atol=0)
   torch.testing.assert_close(logits,base_logits,rtol=0,atol=0)
 def test_per_head_residual_matches_independent_prefix_attention(self):
  base,_=self.a(self.x,self.x,self.x,self.mask)
  prefix_only,_=self.a(self.x,self.p[None].expand(2,-1,-1),self.p[None].expand(2,-1,-1),None)
  full,_=self.a(self.x,self.x,self.x,self.mask,extra_kv=self.p,extra_kv_gate=torch.ones(2))
  torch.testing.assert_close(full,base+prefix_only,rtol=1e-5,atol=1e-6)
 def test_causality_and_padding_are_preserved(self):
  x2=self.x.clone();x2[:,3:]+=100
  gate=torch.tensor([0.3,-0.2])
  a,_=self.a(self.x,self.x,self.x,self.mask,extra_kv=self.p,extra_kv_gate=gate)
  b,_=self.a(x2,x2,x2,self.mask,extra_kv=self.p,extra_kv_gate=gate)
  torch.testing.assert_close(a[:,:3],b[:,:3],rtol=0,atol=0)
  mask=self.mask.clone();mask[:,:,4]=True
  b,_=self.a(self.x,x2,x2,mask,extra_kv=self.p,extra_kv_gate=gate)
  x3=x2.clone();x3[:,4]+=10000
  c,_=self.a(self.x,x3,x3,mask,extra_kv=self.p,extra_kv_gate=gate)
  torch.testing.assert_close(b,c,rtol=0,atol=0)
 def test_gate_then_prefix_receive_gradients(self):
  raw=nn.Parameter(torch.zeros(2));opt=torch.optim.SGD([raw,self.p],lr=0.01)
  y,_=self.a(self.x,self.x,self.x,self.mask,extra_kv=self.p,extra_kv_gate=raw.tanh())
  (y**2).mean().backward()
  self.assertGreater(raw.grad.abs().sum().item(),0)
  self.assertEqual(self.p.grad.abs().max().item(),0)
  opt.step();opt.zero_grad()
  y,_=self.a(self.x,self.x,self.x,self.mask,extra_kv=self.p,extra_kv_gate=raw.tanh())
  (y**2).mean().backward()
  self.assertGreater(self.p.grad.abs().sum().item(),0)
  self.assertTrue(torch.isfinite(raw.grad).all())
 def test_gate_without_prefix_and_wrong_head_count_fail(self):
  with self.assertRaises(ValueError):self.a(self.x,self.x,self.x,self.mask,extra_kv_gate=torch.zeros(2))
  with self.assertRaises(ValueError):self.a(self.x,self.x,self.x,self.mask,extra_kv=self.p,extra_kv_gate=torch.zeros(3))
 def test_layer_gate_serializes_and_missing_gate_fails_strict(self):
  kw=dict(use_proto_prefix=True,proto_prefix_len=4,proto_init=self.p.detach(),use_proto_prefix_gate=True)
  layer=DecoderLayer(12,12,12,12,2,None,**kw)
  self.assertEqual(layer.prefix_gate.numel(),2);self.assertEqual(layer.prefix_gate.abs().max().item(),0)
  sd=layer.state_dict();self.assertIn('prefix_gate',sd)
  broken={k:v for k,v in sd.items() if k!='prefix_gate'}
  with self.assertRaises(RuntimeError):layer.load_state_dict(broken,strict=True)
  clone=DecoderLayer(12,12,12,12,2,None,**kw);clone.load_state_dict(sd)
  self.assertTrue(torch.equal(clone.prefix_gate,layer.prefix_gate))
 def test_gate_requires_prefix_and_finite_initializer(self):
  with self.assertRaises(ValueError):DecoderLayer(12,12,12,12,2,None,use_proto_prefix_gate=True)
  with self.assertRaises(ValueError):DecoderLayer(12,12,12,12,2,None,use_proto_prefix=True,use_proto_prefix_gate=True,proto_prefix_gate_init=float('nan'))
 def test_checkpoint_declares_gated_mode(self):
  import tempfile
  from types import SimpleNamespace
  from src.models.CASE.model import CASE
  root=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory(dir=root/'scratch') as tmp:
   stub=SimpleNamespace(model_dir=tmp,use_proto_prefix=True,use_proto_prefix_gate=True,
      optimizer=SimpleNamespace(state_dict=lambda:{}),
      state_dict=lambda:{'decoder.dec.0.prefix_embed':self.p.detach(),'decoder.dec.0.prefix_gate':torch.zeros(2)})
   with contextlib.redirect_stdout(io.StringIO()):CASE.save_model(stub,1.0,7,tag='composite_best')
   saved=torch.load(next(Path(tmp).glob('CASE_*')),map_location='cpu')
   self.assertEqual(saved['prefix_attention_mode'],'separate_softmax_residual')
if __name__=='__main__':unittest.main()
