#!/usr/bin/env python
"""Validate every stored frame and the complete exported verified-demo collections."""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image


def run(study,out):
    report=json.loads((out/'validation.json').read_text())
    manifest=json.loads((study/'turbovla_afa/manifest.json').read_text())
    source=json.loads((study/'source.json').read_text())
    episodes=[(ep['episode_index'],ep['length']) for ep in manifest['episodes']]
    assert [eid for eid,n in episodes]==source['episode_indices']
    assert sum(n for eid,n in episodes)==source['expected_frames']
    errors={key:0. for key in ['routing_sum','keypoint_reconstruction','fused_sum','dino_row_sum','act_row_sum','act_weight_reconstruction',
        'dino_last_agreement','language_temporal_change','queries_temporal_change','transformer_replay']}
    index=[]
    y,x=np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing='ij');grid=np.stack([x.ravel(),y.ravel()],-1)
    for eid,n in episodes:
        first=None
        for t in range(n):
            paths=[study/kind/f'ep{eid:03d}'/f'f{t:06d}.npz' for kind in ['turbovla_afa','turbovla']]
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                for kind,path,data in zip(['turbovla_afa','turbovla'],paths,[a,b]):
                    assert int(data['frame_index'])==t and int(data['episode_index'])==eid
                    assert data['predicted_actions'].shape==(12,6)
                    assert data['dino_cls_all'].shape==(2,12,12,261)
                    assert float(data['hook_error'])==0
                    for key in data.files:
                        value=data[key]
                        if value.dtype.kind in 'fc':assert np.isfinite(value).all(),(kind,eid,t,key)
                    errors['dino_row_sum']=max(errors['dino_row_sum'],float(np.abs(data['dino_cls_all'].sum(-1)-1).max()))
                    errors['act_row_sum']=max(errors['act_row_sum'],float(np.abs(data['act_all'].sum(-1)-1).max()))
                    mean=data['act_all'][-1].mean((0,1))
                    k=16 if kind=='turbovla_afa' else 256
                    np.testing.assert_allclose(mean[:2*k].reshape(2,k),data['act_weights'],rtol=1e-6,atol=1e-7)
                    errors['act_weight_reconstruction']=max(errors['act_weight_reconstruction'],float(np.abs(mean[:2*k].reshape(2,k)-data['act_weights']).max()))
                    weights=data['dino_cls_all'][:,-1,:,5:].mean(1)
                    weights/=weights.sum(-1,keepdims=True)
                    errors['dino_last_agreement']=max(errors['dino_last_agreement'],float(np.abs(weights.reshape(2,16,16)-data['dino_cls']).max()))
                    index.append(dict(policy=kind,episode=eid,frame=t,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                for key in ['rgb','state','recorded_action','timestamp','words']:
                    np.testing.assert_array_equal(a[key],b[key])
                errors['routing_sum']=max(errors['routing_sum'],float(np.abs(a['routing'].sum(1)-1).max()))
                errors['keypoint_reconstruction']=max(errors['keypoint_reconstruction'],float(np.abs(a['attention'].reshape(2,16,256).astype(float)@grid-a['keypoints']).max()))
                errors['fused_sum']=max(errors['fused_sum'],float(np.abs(a['geometry']+a['semantics']-a['fused']).max()))
                errors['transformer_replay']=max(errors['transformer_replay'],float(a['afa_transformer_replay_error']))
                if first is None:first={key:a[key] for key in ['language','queries_after']}
                errors['language_temporal_change']=max(errors['language_temporal_change'],float(np.abs(a['language']-first['language']).max()))
                errors['queries_temporal_change']=max(errors['queries_temporal_change'],float(np.abs(a['queries_after']-first['queries_after']).max()))
            if t%100==0:print('Verified raw archives',eid,t,'/',n,flush=True)
    for key,value in errors.items():assert value<(2e-5 if key=='transformer_replay' else 3e-6),(key,value)
    with (out/'archive_index.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(index[0]),lineterminator='\n');writer.writeheader();writer.writerows(index)
    pngs=list((out/'clean').rglob('*.png'))+list((out/'charts').glob('*.png'))+list((out/'minimal_labels').rglob('*.png'))
    for path in pngs:
        with Image.open(path) as image:image.verify()
        assert path.with_suffix('.pdf').exists()
        info=subprocess.check_output(['pdfinfo',str(path.with_suffix('.pdf'))],text=True)
        assert re.search(r'Pages:\s+1\b',info)
    for eid,n in episodes:
        files=list((out/'viewer/data'/f'ep{eid:03d}').glob('*.js'));assert len(files)==n
        for t in range(n):
            text=(out/'viewer/data'/f'ep{eid:03d}'/f'f{t:06d}.js').read_text()
            d=json.loads(text[len('window.loadDiagnosticFrame('):-3])
            assert d['episode']==eid and d['frame']==t
            for key,encoded in d['arrays'].items():
                values=np.frombuffer(base64.b64decode(encoded),dtype='<f4');assert np.isfinite(values).all()
            for url in d['rgb']:
                with Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))) as image:
                    assert image.size==(256,192);image.verify()
    selection=json.loads((out/'selection.json').read_text())
    for s in selection['selection']:
        assert len(s['frames'])==12 and len(set(s['frames']))==12 and s['frames']==sorted(s['frames'])
    report.update(raw_archives_verified=len(index),all_raw_arrays_finite=True,paired_rgb_state_actions_identical=True,
        detailed_errors=errors,png_pdf_pairs_verified=len(pngs),viewer_embedded_frames_verified=sum(n for _,n in episodes),all_viewer_rgb_lossless=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Validation passed',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)
