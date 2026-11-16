#!/usr/bin/env python
"""Evaluate every observation of the 12 selected episodes; store compact diagnostics."""

import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from tokenizer_qualitative import sha256


class VideoReader:
    """Sequential nearest-PTS decoding, preserving episode offsets."""
    def __init__(self, path, start):
        import av
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.container.seek(max(0, int(start/self.stream.time_base)), stream=self.stream, backward=True)
        self.frames = self.container.decode(self.stream)
        self.previous = None
        self.current = self._next()

    def _next(self):
        for frame in self.frames:
            if frame.pts is not None:
                return float(frame.pts*self.stream.time_base), frame
        return None

    def read(self, timestamp, fps):
        while self.current is not None and self.current[0] < timestamp:
            self.previous, self.current = self.current, self._next()
        candidates = [x for x in (self.previous, self.current) if x is not None]
        nearest = min(candidates, key=lambda x: abs(x[0]-timestamp))
        if abs(nearest[0]-timestamp) > 1.01/fps:
            raise ValueError(f"Video out of sync: {timestamp}, nearest {nearest[0]}")
        return nearest[1].to_ndarray(format="rgb24"), nearest[0]

    def close(self):
        self.container.close()


class Recorder:
    def __init__(self, core, policy_type):
        import torch
        from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb, eager_attention_forward
        self.active, self.data, self.eager_error = True, {}, None
        self.core, self.policy_type = core, policy_type
        self.torch, self.rope, self.eager = torch, apply_rotary_pos_emb, eager_attention_forward
        last = [m for m in core.vision_encoder.backbone.modules() if m.__class__.__name__ == 'DINOv3ViTAttention'][-1]
        self.handles = [last.register_forward_pre_hook(self.dino, with_kwargs=True),
            core.action_head.decoder.decoder.layers[-1].multihead_attn.register_forward_hook(self.act, with_kwargs=True)]
        if policy_type == 'turbovla_afa':
            for view, pool in enumerate(core.afa_tokenizers):
                pool.record_inference_diagnostics = True
                self.handles.append(pool.register_forward_hook(self.pool(view)))

    def dino(self, module, inputs, kwargs):
        if not self.active:
            return
        x = inputs[0] if inputs else kwargs['hidden_states']
        b, n, _ = x.shape
        q = module.q_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
        k = module.k_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
        q,k = self.rope(q,k,*kwargs['position_embeddings'])
        mask = kwargs.get('attention_mask')
        logits = q[:,:,:1] @ k.transpose(-2,-1) * module.scaling
        if mask is not None:
            logits = logits + mask[...,:1,:]
        weights = logits.softmax(-1)
        if self.eager_error is None:
            v = module.v_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
            _, reference = self.eager(module,q,k,v,mask,scaling=module.scaling,dropout=0)
            self.eager_error = float((reference[:,:,:1]-weights).abs().max())
            if self.eager_error > 2e-6:
                raise ValueError(f'CLS replay mismatch: {self.eager_error}')
        prefix = self.core.vision_encoder.prefix_tokens
        patches = weights[0,:,0,prefix:]
        p = patches.mean(0)
        self.data.setdefault('dino_cls',[]).append((p/p.sum()).view(16,16).cpu().numpy())
        self.data.setdefault('dino_patch_mass',[]).append(float(patches.sum(-1).mean()))

    def act(self,module,inputs,kwargs,output):
        if self.active:
            _, weights = module.forward(*inputs,**dict(kwargs,need_weights=True,average_attn_weights=False))
            self.data['act'] = weights[0].mean((0,1)).cpu().numpy()

    def pool(self, view):
        def hook(module,inputs,output):
            if self.active:
                for key,value in [('attention',module.latest_attention),('routing',module.latest_query_softmax),('keypoints',output[0])]:
                    self.data.setdefault(key,{})[view] = value[0].detach().cpu().numpy()
        return hook

    def take(self):
        d = self.data
        result = {key:np.asarray(d[key],dtype=np.float32) for key in ['dino_cls','dino_patch_mass']}
        maps,masses,slot_weights = [],[],[]
        slots = 16 if self.policy_type == 'turbovla_afa' else 256
        for view in (0,1):
            w = d['act'][view*slots:(view+1)*slots]
            masses.append(w.sum())
            slot_weights.append(w)
            if slots == 16:
                p = d['attention'][view].reshape(16,-1)
                p = p/np.maximum(p.sum(-1,keepdims=True),1e-12)
                maps.append(((w/w.sum()) @ p).reshape(16,16))
            else:
                maps.append((w/w.sum()).reshape(16,16))
        result.update(act_map=np.asarray(maps,dtype=np.float32),act_mass=np.asarray(masses,dtype=np.float32),
                      act_weights=np.asarray(slot_weights,dtype=np.float32))
        if slots == 16:
            for key in ['attention','routing','keypoints']:
                result[key] = np.stack([d[key][v] for v in (0,1)])
        self.data = {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def episodes(source):
    manifest = json.loads(source.read_text())
    selected = {}
    for r in manifest['records']:
        key = (r['group'],r['task_id'],r['episode'])
        if key not in selected:
            selected[key] = r
    return manifest,list(selected.values())


def run(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original, selected = episodes(args.manifest)
    saved = json.loads((args.checkpoint/'config.json').read_text())
    kind = saved['type']
    importlib.import_module(f'lerobot_policy_{kind}')
    root = args.output/kind
    root.mkdir(parents=True,exist_ok=True)
    meta = dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256(args.checkpoint/'model.safetensors'),
        config=saved,device=args.device,precision='fp32',torch_version=torch.__version__,batch_size=1,
        full_episode_coverage=True,selection='All frames of both selected verified demonstrations and both failures per task',
        episodes=selected,sources=original['sources'],display_crop_xyxy=[0,32,256,224],
        crop_applied_to_model_input=False,limit_frames=args.limit_frames)
    path = root/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ['checkpoint_sha256','device','episodes','limit_frames']:
            if old[key] != meta[key]:
                raise ValueError(f'Existing output differs in {key}')
    path.write_text(json.dumps(meta,indent=2))
    if all((root/f"{r['group']}_task{r['task_id']}_ep{r['episode']:03d}.npz").exists() for r in selected):
        print('All episode archives already complete.',flush=True)
        return
    config = PreTrainedConfig.from_pretrained(args.checkpoint,cli_overrides=[
        '--device=cpu','--inference_dtype=null','--use_amp=false','--vision.compute_precision=fp32','--interaction.compute_precision=fp32'])
    policy = get_policy_class(kind).from_pretrained(args.checkpoint,config=config,local_files_only=True,strict=True).eval().to(args.device)
    pre,post = make_pre_post_processors(config,pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor':{'device':'cpu'}},
        postprocessor_overrides={'device_processor':{'device':'cpu'}})
    rec = Recorder(policy.model,kind)
    report = dict(episodes=[],expected_frames=sum(r['episode_metadata']['length'] for r in selected),
        completed_frames=0,hook_checks=0,hook_max_error=0,max_video_sync_error_s=0)
    if (root/'coverage.json').exists():
        report.update(json.loads((root/'coverage.json').read_text()))
        report['completed_frames']=0
    tables = {}
    newly_completed=0
    try:
        with torch.inference_mode():
            for ep_index,r in enumerate(selected):
                name = f"{r['group']}_task{r['task_id']}_ep{r['episode']:03d}"
                output = root/f'{name}.npz'
                if output.exists():
                    with np.load(output) as old:
                        report['completed_frames'] += len(old['frame_index'])
                    continue
                info = json.loads((Path(r['root'])/'meta/info.json').read_text())
                ep = r['episode_metadata']
                data_path = Path(r['root'])/info['data_path'].format(chunk_index=ep['data/chunk_index'],file_index=ep['data/file_index'])
                if data_path not in tables:
                    tables[data_path] = pq.read_table(data_path).to_pandas()
                table = tables[data_path]
                table = table[table.episode_index==r['episode']].sort_values('frame_index')
                np.testing.assert_array_equal(np.asarray(table.frame_index),np.arange(ep['length']))
                if args.limit_frames:
                    table = table.iloc[:args.limit_frames]
                videos = []
                for key in config.camera_keys:
                    video_path = Path(r['root'])/info['video_path'].format(video_key=key,
                        chunk_index=ep[f'videos/{key}/chunk_index'],file_index=ep[f'videos/{key}/file_index'])
                    videos.append(VideoReader(video_path,ep[f'videos/{key}/from_timestamp']))
                buffers = {}
                started = time.monotonic()
                for idx,(_,row) in enumerate(table.iterrows()):
                    raw = {'task':[r['task']],'observation.state':torch.tensor(np.asarray(row['observation.state'],dtype=np.float32))[None]}
                    errors=[]
                    for v,key in enumerate(config.camera_keys):
                        t = ep[f'videos/{key}/from_timestamp']+float(row.timestamp)
                        rgb,actual = videos[v].read(t,r['fps'])
                        errors.append(abs(t-actual))
                        raw[key] = torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255
                    cpu = pre(raw)
                    batch = {k:value.to(args.device) if isinstance(value,torch.Tensor) else value for k,value in cpu.items()}
                    rec.active=True
                    prediction = policy.predict_action_chunk(batch)
                    d = rec.take()
                    if idx in {0,len(table)//2,len(table)-1}:
                        rec.active=False
                        other = policy.predict_action_chunk(batch)
                        error = float((prediction-other).abs().max())
                        if error > 1e-6:
                            raise ValueError(f'Instrumentation changed prediction: {error}')
                        report['hook_checks']+=1
                        report['hook_max_error']=max(error,report['hook_max_error'])
                    d['predicted_actions'] = post(prediction).cpu().numpy()[0]
                    d['frame_index'] = np.asarray(row.frame_index,dtype=np.int64)
                    d['timestamp'] = np.asarray(row.timestamp,dtype=np.float32)
                    d['state'] = np.asarray(row['observation.state'],dtype=np.float32)
                    d['video_sync_error_s'] = np.asarray(errors,dtype=np.float64)
                    if kind=='turbovla_afa':
                        images=[]
                        for key in config.camera_keys:
                            mean=torch.tensor(config.image_mean).view(1,3,1,1)
                            std=torch.tensor(config.image_std).view(1,3,1,1)
                            rgb=(cpu[key]*std+mean)[0].permute(1,2,0).clamp(0,1).numpy()
                            images.append(np.round(rgb[32:224]*255).astype(np.uint8))
                        d['rgb']=np.stack(images)
                    for key,value in d.items():
                        if np.asarray(value).dtype.kind in 'fc' and not np.isfinite(value).all():
                            raise ValueError(f'Nonfinite {key} in {name}, frame {idx}')
                        buffers.setdefault(key,[]).append(value)
                    report['max_video_sync_error_s']=max(report['max_video_sync_error_s'],max(errors))
                    if (idx+1)%50==0 or idx+1==len(table):
                        print(f'{kind} [{ep_index+1}/{len(selected)}] {name} {idx+1}/{len(table)} frames, {(idx+1)/(time.monotonic()-started):.1f} frames/s',flush=True)
                for video in videos:
                    video.close()
                arrays={key:np.stack(values) for key,values in buffers.items()}
                temp=output.with_suffix('.npz.tmp')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream,**arrays)
                temp.replace(output)
                report['completed_frames']+=len(table)
                report['episodes'].append(dict(name=name,frames=len(table),expected=ep['length'],seconds=time.monotonic()-started))
                report['cls_eager_max_error']=rec.eager_error
                if args.device.startswith('cuda'):
                    report['peak_allocated_cuda_MiB']=torch.cuda.max_memory_allocated(args.device)/1024**2
                (root/'coverage.json').write_text(json.dumps(report,indent=2))
                newly_completed+=1
                if args.max_episodes and newly_completed>=args.max_episodes:
                    break
    finally:
        rec.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit-frames',type=int)
    p.add_argument('--max-episodes',type=int,help='Process this many new episodes per invocation; safe to resume')
    run(p.parse_args())


if __name__=='__main__':
    main()
