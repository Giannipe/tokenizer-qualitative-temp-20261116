#!/usr/bin/env python
"""Baseline replay using full-matrix QK for numerically sharp DINO heads.

The optimized single-row GEMM can differ from the full GEMM in a very sharp
early baseline head. Keep production SDPA intact and match the eager matrix
shape exactly for the diagnostic computation. The AFA recording passed the
single-row check; this wrapper is explicitly pinned in baseline provenance.
"""
import argparse
import json
from pathlib import Path
import analyze_successgrasp_diagnostics as base
from tokenizer_qualitative import sha256


def full_layer_hook(self, layer):
    def hook(module, inputs, kwargs):
        if not self.active:
            return
        x=inputs[0] if inputs else kwargs['hidden_states']
        b,n,_=x.shape
        q=module.q_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
        k=module.k_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
        q,k=self.rope(q,k,*kwargs['position_embeddings'])
        mask=kwargs.get('attention_mask')
        scores=(q@k.transpose(-2,-1))*module.scaling
        if mask is not None:
            scores=scores+mask
        weights=scores.softmax(-1)[:,:,:1]
        if layer not in self.layer_errors:
            v=module.v_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
            _,reference=self.eager(module,q,k,v,mask,scaling=module.scaling,dropout=0)
            error=float((weights-reference[:,:,:1]).abs().max())
            if error>2e-6:
                raise ValueError(('Full QK differs from eager',layer,error))
            self.layer_errors[layer]=error
        self.extra.setdefault(f'dino_layer_{layer}',[]).append(self.array(weights[0,:,0]))
    return hook


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--limit-frames',type=int)
    args=parser.parse_args()
    base.CompleteRecorder.layer_hook=full_layer_hook
    root=args.study/'turbovla'
    root.mkdir(exist_ok=True)
    (root/'fullqk_provenance.json').write_text(json.dumps(dict(wrapper_sha256=sha256(Path(__file__)),
        base_source_sha256=sha256(Path(base.__file__)),diagnostic='full-matrix QK matching eager; all DINO layers; production SDPA unchanged'),indent=2))
    base.run(args)
