#!/usr/bin/env python
"""Full-frame, full-slot paired diagnostics of a pinned LeRobot rollout."""
import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from analyze_full_episodes import Recorder, VideoReader
from tokenizer_qualitative import read_episodes, sha256


class CompleteRecorder(Recorder):
    def __init__(self, core, kind):
        self.extra = {}
        self.layer_errors = {}
        super().__init__(core, kind)
        modules = [m for m in core.vision_encoder.backbone.modules()
                   if m.__class__.__name__ == 'DINOv3ViTAttention']
        for layer, module in enumerate(modules):
            self.handles.append(module.register_forward_pre_hook(self.layer_hook(layer), with_kwargs=True))
        self.handles.append(core.vision_encoder.register_forward_hook(self.output_hook('dino_features')))
        for layer, module in enumerate(core.action_head.decoder.decoder.layers):
            self.handles.append(module.multihead_attn.register_forward_hook(self.mha_hook(f'act_layer_{layer}'), with_kwargs=True))
        if kind == 'turbovla_afa':
            for view, pool in enumerate(core.afa_tokenizers):
                self.handles.append(pool.register_forward_hook(self.pool_details(view)))
                self.handles.append(pool.coord_encoder.register_forward_hook(self.output_hook(f'geometry_{view}')))
                self.handles.append(pool.semantic_encoder.register_forward_hook(self.semantic_hook(view)))
                conditioning = pool.task_query_cross_attention
                self.handles.append(conditioning.register_forward_hook(self.query_hook(view)))
                self.handles.append(conditioning.cross_attention.register_forward_hook(self.mha_hook(f'language_{view}'), with_kwargs=True))
            if core.afa_transformer is not None:
                self.handles.append(core.afa_transformer.register_forward_hook(self.transformer_hook))

    def array(self, x):
        return x.detach().float().cpu().numpy()

    def output_hook(self, name):
        def hook(module, inputs, output):
            if self.active:
                self.extra[name] = self.array(output)[0]
        return hook

    def layer_hook(self, layer):
        def hook(module, inputs, kwargs):
            if not self.active:
                return
            x = inputs[0] if inputs else kwargs['hidden_states']
            b, n, _ = x.shape
            q = module.q_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
            k = module.k_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
            q,k = self.rope(q,k,*kwargs['position_embeddings'])
            mask = kwargs.get('attention_mask')
            scores = (q[:,:,:1] @ k.transpose(-2,-1))*module.scaling
            if mask is not None:
                scores = scores+mask[...,:1,:]
            weights = scores.softmax(-1)
            if layer not in self.layer_errors:
                v = module.v_proj(x).view(b,n,module.num_heads,module.head_dim).transpose(1,2)
                _, reference = self.eager(module,q,k,v,mask,scaling=module.scaling,dropout=0)
                error = float((weights-reference[:,:,:1]).abs().max())
                if error > 2e-6:
                    raise ValueError(('DINO eager mismatch',layer,error))
                self.layer_errors[layer] = error
            self.extra.setdefault(f'dino_layer_{layer}',[]).append(self.array(weights[0,:,0]))
        return hook

    def mha_hook(self, name):
        def hook(module, inputs, kwargs, output):
            if self.active:
                _, weights = module.forward(*inputs,**dict(kwargs,need_weights=True,average_attn_weights=False))
                self.extra[name] = self.array(weights)[0]
                if name.startswith('language_'):
                    self.extra[name+'_padding'] = kwargs['key_padding_mask'][0].detach().cpu().numpy()
        return hook

    def semantic_hook(self, view):
        def hook(module, inputs, output):
            if self.active:
                self.extra[f'pooled_values_{view}'] = self.array(inputs[0])[0]
                self.extra[f'semantics_{view}'] = self.array(output)[0]
        return hook

    def query_hook(self, view):
        def hook(module, inputs, output):
            if self.active:
                self.extra[f'queries_before_{view}'] = self.array(inputs[0])[0]
                self.extra[f'queries_after_{view}'] = self.array(output)[0]
        return hook

    def pool_details(self, view):
        def hook(module, inputs, output):
            if self.active:
                self.extra[f'fused_{view}'] = self.array(output[1])[0]
        return hook

    def transformer_hook(self, module, inputs, output):
        if not self.active:
            return
        # Replay on a separate tensor; do not hook encoder-layer children, which
        # would disable PyTorch's fused production TransformerEncoder fast path.
        tokens, coords = inputs
        tokens = tokens.flatten(0,1)
        x = tokens if module.drop_keypoints else self.torch.cat([coords.flatten(0,1),tokens],-1)
        x = module.token_embedding(x)
        for i,layer in enumerate(module.transformer.layers):
            q = layer.norm1(x) if layer.norm_first else x
            _, weights = layer.self_attn.forward(q,q,q,need_weights=True,average_attn_weights=False)
            self.extra[f'afa_self_layer_{i}'] = self.array(weights)
            x = layer(x)
        if module.transformer.norm is not None:
            x = module.transformer.norm(x)
        error = float((x-output.flatten(0,1)).abs().max())
        if error > 2e-5:
            raise ValueError(('AFA transformer replay mismatch',error))
        self.extra['afa_transformer_replay_error'] = np.asarray(error)
        self.extra['post_transformer'] = self.array(output)[0]

    def take(self):
        d = super().take()
        e = self.extra
        d['dino_features'] = e['dino_features']
        d['dino_cls_all'] = np.stack([e[f'dino_layer_{i}'] for i in range(len(self.layer_errors))],axis=1)
        d['act_all'] = np.stack([e[f'act_layer_{i}'] for i in range(3)])
        if self.policy_type == 'turbovla_afa':
            for key in ['geometry','semantics','pooled_values','fused','queries_before','queries_after','language','language_padding']:
                names = [f'{key}_{v}' for v in (0,1)] if key!='language_padding' else [f'language_{v}_padding' for v in (0,1)]
                d[key] = np.stack([e[name] for name in names])
            d['post_transformer'] = e['post_transformer']
            d['afa_self_attention'] = np.stack([e[name] for name in sorted(e) if name.startswith('afa_self_layer_')],axis=1)
            d['afa_transformer_replay_error'] = e['afa_transformer_replay_error']
        self.extra = {}
        return d


def run(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class,make_pre_post_processors
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=json.loads((args.study/'source.json').read_text())
    dataset=Path(source['local_root'])
    info=json.loads((dataset/'meta/info.json').read_text())
    episodes=read_episodes(dataset)
    episodes=[{k:v for k,v in ep.items() if not k.startswith('stats/')} for ep in episodes]
    saved=json.loads((args.checkpoint/'config.json').read_text());kind=saved['type']
    importlib.import_module(f'lerobot_policy_{kind}')
    root=args.study/kind;root.mkdir(exist_ok=True)
    manifest=dict(schema='successgrasp_complete_v1',checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint/'model.safetensors'),
        config=saved,source=source,episodes=episodes,device=args.device,precision='fp32',batch_size=1,torch_version=torch.__version__,
        diagnostic_source_sha256=sha256(Path(__file__)),all_frames=True,all_slots=True,display_crop_xyxy=[0,32,256,224])
    config=PreTrainedConfig.from_pretrained(args.checkpoint,cli_overrides=['--device=cpu','--inference_dtype=null','--use_amp=false','--vision.compute_precision=fp32','--interaction.compute_precision=fp32'])
    policy=get_policy_class(kind).from_pretrained(args.checkpoint,config=config,local_files_only=True,strict=True).eval().to(args.device)
    pre,post=make_pre_post_processors(config,pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor':{'device':'cpu'}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
    rec=CompleteRecorder(policy.model,kind)
    report=dict(expected_frames=sum(ep['length'] for ep in episodes),completed_frames=0,hook_checks=0,hook_max_error=0.,max_video_sync_error_s=0.,episodes=[])
    if (root/'manifest.json').exists():
        previous=json.loads((root/'manifest.json').read_text())
        for key in ['schema','checkpoint_sha256','diagnostic_source_sha256']:
            if previous[key]!=manifest[key]:raise ValueError(('Output differs',key))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    processed=0
    try:
        with torch.inference_mode():
            for ep in episodes:
                index=ep['episode_index'];folder=root/f'ep{index:03d}';folder.mkdir(exist_ok=True)
                table=pq.read_table(dataset/info['data_path'].format(chunk_index=ep['data/chunk_index'],file_index=ep['data/file_index'])).to_pandas()
                table=table[table.episode_index==index].sort_values('frame_index')
                np.testing.assert_array_equal(table.frame_index,np.arange(ep['length']))
                readers=[]
                for key in config.camera_keys:
                    video=dataset/info['video_path'].format(video_key=key,chunk_index=ep[f'videos/{key}/chunk_index'],file_index=ep[f'videos/{key}/file_index'])
                    readers.append(VideoReader(video,ep[f'videos/{key}/from_timestamp']))
                task=ep['tasks'][0];words=policy.model.text_encoder.tokenizer.convert_ids_to_tokens(policy.model.text_encoder.tokenizer(task,truncation=True,max_length=config.text.max_length)['input_ids'])
                started=time.monotonic()
                for _,row in table.iterrows():
                    destination=folder/f'f{row.frame_index:06d}.npz'
                    if destination.exists():
                        report['completed_frames']+=1
                        with np.load(destination) as old:
                            report['hook_checks']+=1;report['hook_max_error']=max(report['hook_max_error'],float(old['hook_error']))
                            report['max_video_sync_error_s']=max(report['max_video_sync_error_s'],float(old['video_sync_error_s'].max()))
                        continue
                    raw={'task':[task],'observation.state':torch.tensor(np.asarray(row['observation.state'],dtype=np.float32))[None]}
                    errors=[]
                    for view,key in enumerate(config.camera_keys):
                        t=ep[f'videos/{key}/from_timestamp']+float(row.timestamp)
                        rgb,pts=readers[view].read(t,info['fps']);errors.append(abs(t-pts))
                        raw[key]=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255
                    cpu=pre(raw);batch={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
                    rec.active=True;prediction=policy.predict_action_chunk(batch);d=rec.take()
                    rec.active=False;control=policy.predict_action_chunk(batch)
                    error=float((prediction-control).abs().max())
                    if error>1e-6:raise ValueError(('Diagnostic changed prediction',error))
                    report['hook_checks']+=1;report['hook_max_error']=max(report['hook_max_error'],error)
                    images=[]
                    for key in config.camera_keys:
                        mean=torch.tensor(config.image_mean).view(1,3,1,1);std=torch.tensor(config.image_std).view(1,3,1,1)
                        image=((cpu[key]*std+mean)[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
                        images.append(image[32:224])
                    d.update(rgb=np.stack(images),predicted_actions=post(prediction).cpu().numpy()[0],recorded_action=np.asarray(row.action,dtype=np.float32),
                        state=np.asarray(row['observation.state'],dtype=np.float32),timestamp=np.asarray(row.timestamp),frame_index=np.asarray(row.frame_index),
                        episode_index=np.asarray(index),words=np.asarray(words),video_sync_error_s=np.asarray(errors),hook_error=np.asarray(error))
                    for key,value in d.items():
                        if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError(('Nonfinite',key))
                    temporary=destination.with_suffix('.npz.tmp')
                    with temporary.open('wb') as stream:np.savez_compressed(stream,**d)
                    temporary.replace(destination)
                    processed+=1;report['completed_frames']+=1;report['max_video_sync_error_s']=max(report['max_video_sync_error_s'],max(errors))
                    if processed%25==0:print(kind,index,row.frame_index,'fps',round(processed/(time.monotonic()-started),2),flush=True)
                    if args.limit_frames and processed>=args.limit_frames:
                        break
                for reader in readers:reader.close()
                report['episodes'].append(dict(episode=index,expected_frames=ep['length'],saved_frames=len(list(folder.glob('f*.npz')))))
                report['dino_layer_eager_errors']=rec.layer_errors
                report['peak_cuda_MiB']=torch.cuda.max_memory_allocated(args.device)/1024**2
                (root/'coverage.json').write_text(json.dumps(report,indent=2))
                if args.limit_frames and processed>=args.limit_frames:break
    finally:
        rec.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit-frames',type=int)
    run(p.parse_args())
