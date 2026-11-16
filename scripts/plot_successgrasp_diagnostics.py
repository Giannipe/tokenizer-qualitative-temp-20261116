#!/usr/bin/env python
"""Render complete success-grasp diagnostics, with all-frame browser assets."""
import argparse
import base64
import csv
import hashlib
import json
import io
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image, ImageDraw

from select_clean_full_episodes import montage, overlay, rank_episode, save_pair, with_labels
from plot_tokenizer_qualitative import COLORS


def probability(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(x.sum(-1,keepdims=True),1e-12)


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def pca_fit(x):
    x=np.asarray(x,dtype=np.float64)
    mean=x.mean(0);centered=x-mean
    values,vectors=np.linalg.eigh(centered.T@centered/max(len(x)-1,1))
    basis=vectors[:,-3:][:,::-1]
    for i in range(3):
        if basis[np.abs(basis[:,i]).argmax(),i]<0:basis[:,i]*=-1
    score=centered@basis
    bounds=np.quantile(score,[.02,.98],axis=0)
    return dict(mean=mean,basis=basis,bounds=bounds,explained=float(values[-3:].sum()/max(values.sum(),1e-12)))


def pca_color(x,model):
    score=(x-model['mean'])@model['basis']
    return np.clip((score-model['bounds'][0])/np.maximum(np.diff(model['bounds'],axis=0)[0],1e-12),.06,.94)


def up(x,nearest=False):
    mode=Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    x=np.asarray(x)
    if x.ndim==2:
        return np.asarray(Image.fromarray(x.astype(np.float32)).resize((256,256),mode))[32:224]
    return np.stack([up(x[...,i],nearest) for i in range(x.shape[-1])],-1)


def points(rgb,coords,colors,attention=None,ellipses=False,ids=False):
    # Coordinate grid endpoints correspond to patch centers 7.5 and 247.5.
    image=Image.fromarray(rgb).convert('RGBA');layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
    positions=coords*120+np.array([127.5,95.5])
    for k,(x,y) in enumerate(positions):
        color=tuple(np.round(np.clip(colors[k],0,1)*255).astype(int))
        if ellipses and attention is not None:
            p=probability(attention[k].reshape(-1))
            gy,gx=np.meshgrid(np.arange(16)*16+7.5-32,np.arange(16)*16+7.5,indexing='ij')
            grid=np.stack([gx.ravel(),gy.ravel()],-1);delta=grid-np.array([x,y])
            cov=(delta*p[:,None]).T@delta
            values,vectors=np.linalg.eigh(cov)
            theta=np.linspace(0,2*np.pi,65)
            curve=(vectors@np.diag(np.sqrt(np.maximum(values,0)))@np.stack([np.cos(theta),np.sin(theta)])).T+[x,y]
            draw.line([tuple(v) for v in curve],fill=color+(125,),width=1)
        draw.ellipse([x-3.2,y-3.2,x+3.2,y+3.2],fill=color+(245,),outline=(255,255,255,255),width=1)
        if ids:draw.text((x+4,y-7),str(k),fill=(255,255,255,255),stroke_width=1,stroke_fill=(0,0,0,255))
    return np.asarray(Image.alpha_composite(image,layer).convert('RGB'))


def blend(rgb,color,alpha=.55):
    return np.round(np.clip((1-alpha)*rgb+alpha*255*color,0,255)).astype(np.uint8)


def colored_routing(rgb,routing):
    r=probability(routing.T).T.reshape(16,16,16)
    winners=r.argmax(0);confidence=(r.max(0)-1/16)/(1-1/16)
    color=up(COLORS[winners],nearest=True)
    alpha=.60*up(np.maximum(confidence,0),nearest=True)[...,None]
    return np.round(rgb*(1-alpha)+color*255*alpha).astype(np.uint8)


def masked_rgb(rgb,routing,k):
    w=up(routing[k].reshape(16,16))[...,None]
    return np.round(np.clip(rgb*w+255*(1-w),0,255)).astype(np.uint8)


def projected_colors(rgb,routing,colors,coords=None):
    r=probability(routing.T).T
    color=up((r.T@colors).reshape(16,16,3))
    result=blend(rgb,color)
    return result if coords is None else points(result,coords,colors)


def summarize(study,out):
    cache=Path('/tmp')/f'{study.name}_summary.npz'
    if cache.exists():
        with np.load(cache) as data:
            return {k:data[k] for k in data.files
                    if not k.endswith(('queries_before','queries_after','pooled_values'))
                    and (not k.startswith('dino_calibration_') or not (out/'pca_models.npz').exists())}
    afa=study/'turbovla_afa';baseline=study/'turbovla'
    metadata=json.loads((afa/'manifest.json').read_text())
    result={};calibration={0:[],1:[]}
    simple=['attention','routing','keypoints','semantics','geometry','fused','pooled_values','post_transformer',
        'queries_before','queries_after','language','language_padding','afa_self_attention','act_weights','act_mass',
        'dino_cls','dino_patch_mass','act_map','rgb','state','timestamp','frame_index','episode_index','predicted_actions',
        'recorded_action','hook_error','video_sync_error_s','afa_transformer_replay_error']
    for ep in metadata['episodes']:
        eid=ep['episode_index'];n=ep['length'];buffers={}
        for t in range(n):
            with np.load(afa/f'ep{eid:03d}'/f'f{t:06d}.npz') as a, np.load(baseline/f'ep{eid:03d}'/f'f{t:06d}.npz') as b:
                for key in simple:buffers.setdefault(key,[]).append(a[key])
                if t==0:result[f'ep{eid}_words']=a['words']
                for key in ['frame_index','episode_index','state','recorded_action','rgb','timestamp']:
                    np.testing.assert_array_equal(a[key],b[key])
                for prefix,d in [('afa',a),('base',b)]:
                    for key in ['dino_cls','dino_patch_mass','act_map','act_mass','predicted_actions','hook_error']:
                        buffers.setdefault(f'{prefix}_{key}',[]).append(d[key])
                    cls=d['dino_cls_all'][...,5:]
                    layer=cls.mean(2);layer=probability(layer)
                    buffers.setdefault(prefix+'_dino_layers',[]).append(layer.reshape(2,12,16,16))
                    buffers.setdefault(prefix+'_dino_heads',[]).append(probability(cls[:,-1]).reshape(2,12,16,16))
                    buffers.setdefault(prefix+'_dino_head_patch_mass',[]).append(cls.sum(-1))
                    act=d['act_all'];k=16 if prefix=='afa' else 256
                    byhead=[];byquery=[];mass=[]
                    for view in (0,1):
                        weights=act[...,view*k:(view+1)*k]
                        maps=probability(d['attention'][view].reshape(16,256)) if k==16 else np.eye(256)
                        mean=weights.mean(2)
                        byhead.append(((probability(mean)@maps) if k==16 else probability(mean)).reshape(3,8,16,16))
                        byquery.append(((probability(weights[-1].mean(0))@maps) if k==16 else probability(weights[-1].mean(0))).reshape(12,16,16))
                        mass.append(mean.sum(-1))
                    buffers.setdefault(prefix+'_act_heads',[]).append(np.stack(byhead))
                    buffers.setdefault(prefix+'_act_queries',[]).append(np.stack(byquery))
                    buffers.setdefault(prefix+'_act_head_mass',[]).append(np.stack(mass))
                    if t%16==0 or t==n-1:
                        features=d['dino_features']
                        for view in (0,1):calibration[view].append(features[view].reshape(16,16,-1)[2:14].reshape(-1,features.shape[-1]))
            if t%100==0:print('Summary',eid,t,'/',n,flush=True)
        for key,values in buffers.items():result[f'ep{eid}_{key}']=np.stack(values)
    for view in (0,1):result[f'dino_calibration_{view}']=np.concatenate(calibration[view])
    np.savez_compressed(cache,**result)
    return result


def metric_tables(data,out,episodes):
    slots=[];frames=[];report=dict(all_arrays_finite=True,max_fusion_error=0.,max_routing_sum_error=0.,max_keypoint_error=0.,hook_error=0.,max_video_sync_error_s=0.)
    y,x=np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing='ij');grid=np.stack([x.ravel(),y.ravel()],-1)
    for eid,n in episodes:
        d={key[len(f'ep{eid}_'):]:v for key,v in data.items() if key.startswith(f'ep{eid}_')}
        np.testing.assert_array_equal(d['frame_index'],np.arange(n))
        report['max_fusion_error']=max(report['max_fusion_error'],float(np.abs(d['geometry']+d['semantics']-d['fused']).max()))
        report['max_routing_sum_error']=max(report['max_routing_sum_error'],float(np.abs(d['routing'].sum(2)-1).max()))
        coords=d['attention'].reshape(n,2,16,256).astype(float)@grid
        report['max_keypoint_error']=max(report['max_keypoint_error'],float(np.abs(coords-d['keypoints']).max()))
        report['hook_error']=max(report['hook_error'],float(d['afa_hook_error'].max()),float(d['base_hook_error'].max()))
        report['max_video_sync_error_s']=max(report['max_video_sync_error_s'],float(d['video_sync_error_s'].max()))
        for view in (0,1):
            rank=rank_episode(d,view)
            for t in range(n):
                p=probability(d['attention'][t,view].reshape(16,256));r=d['routing'][t,view]
                spatial=unit(p-1/256)@unit(p-1/256).T
                semantic=unit(d['semantics'][t,view])@unit(d['semantics'][t,view]).T
                h=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(256)
                for k in range(16):
                    slots.append(dict(episode=eid,frame=t,camera=view,slot=k,entropy=h[k],alive_mass=d['attention'][t,view,k].sum(),
                        visible_mass=p[k].reshape(16,16)[2:14].sum(),act_weight=d['act_weights'][t,view,k],
                        geometry_norm=np.linalg.norm(d['geometry'][t,view,k]),semantic_norm=np.linalg.norm(d['semantics'][t,view,k]),
                        quality=rank['quality'][t,k],x=d['keypoints'][t,view,k,0],y=d['keypoints'][t,view,k,1]))
                mask=~np.eye(16,dtype=bool)
                frames.append(dict(episode=eid,frame=t,camera=view,time=d['timestamp'][t],mean_spatial_entropy=h.mean(),
                    routing_entropy=-(r*np.log(np.maximum(r,1e-30))).sum(0).mean()/np.log(16),
                    spatial_redundancy=spatial[mask].mean(),semantic_similarity=semantic[mask].mean(),
                    afa_act_mass=d['act_mass'][t,view],base_act_mass=d['base_act_mass'][t,view],
                    quality=rank['frame_quality'][t],sharpness=rank['sharpness'][t]))
    for name,rows in [('slot_metrics.csv',slots),('frame_metrics.csv',frames)]:
        with (out/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    report.update(total_frames_per_policy=sum(n for _,n in episodes),camera_views_per_policy=2*sum(n for _,n in episodes),slot_metric_rows=len(slots))
    assert report['max_fusion_error']<1e-6 and report['max_routing_sum_error']<2e-6 and report['max_keypoint_error']<3e-6 and report['hook_error']<1e-6
    return report


def figure_save(fig,path):
    path.parent.mkdir(exist_ok=True,parents=True)
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight',pad_inches=.03,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.03,facecolor='white')
    plt.close(fig)


def heatmap(matrix,path,rows=None,columns=None,vmin=0,vmax=1,cmap='magma'):
    fig,ax=plt.subplots(figsize=(max(4.6,(matrix.shape[1])*.35),max(3.5,matrix.shape[0]*.23)))
    im=ax.imshow(matrix,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest')
    ax.set_yticks(np.arange(matrix.shape[0]),rows if rows is not None else np.arange(matrix.shape[0]),fontsize=7)
    ax.set_xticks(np.arange(matrix.shape[1]),columns if columns is not None else np.arange(matrix.shape[1]),rotation=50 if columns is not None else 0,ha='right',fontsize=7)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.025)
    figure_save(fig,path)


def make_models(data,episodes,out):
    if (out/'pca_models.npz').exists():
        with np.load(out/'pca_models.npz') as z:
            return {f'{name}_{view}':{key:z[f'{name}_{view}_{key}'] for key in ['mean','basis','bounds','explained']}
                    for view in (0,1) for name in ['semantics','geometry','fused','post_transformer','dino']}
    models={};arrays={}
    for view in (0,1):
        for name in ['semantics','geometry','fused','post_transformer']:
            x=np.concatenate([data[f'ep{eid}_{name}'][:,view].reshape(-1,256) for eid,_ in episodes])
            model=pca_fit(x);models[f'{name}_{view}']=model
            for key,value in model.items():arrays[f'{name}_{view}_{key}']=np.asarray(value)
        model=pca_fit(data[f'dino_calibration_{view}']);models[f'dino_{view}']=model
        for key,value in model.items():arrays[f'dino_{view}_{key}']=np.asarray(value)
    np.savez_compressed(out/'pca_models.npz',**arrays)
    return models


def export_figures(study,out,data,models,episodes):
    catalog=[];selections=[]
    def emit(image,name,caption,labels=None,times=None):
        save_pair(image,out/'clean'/name)
        if labels is not None:
            save_pair(with_labels(image,labels,times),out/'minimal_labels'/name)
        catalog.append(dict(name=name,caption=caption,png=f'clean/{name}.png',pdf=f'clean/{name}.pdf'))
    for eid,n in episodes:
        d={key[len(f'ep{eid}_'):]:v for key,v in data.items() if key.startswith(f'ep{eid}_')}
        for view in (0,1):
            camera=['front','wrist'][view];ident=f'ep{eid:03d}_{camera}'
            rank=rank_episode(d,view)
            # Twelve chronological bins; favor focus/readability only within a bin.
            bins=np.array_split(np.arange(n),12)
            selected=[int(group[rank['frame_quality'][group].argmax()]) for group in bins]
            atlas_frames=[selected[i] for i in (0,2,4,6,8,11)]
            selections.append(dict(episode=eid,camera=camera,frames=selected,atlas_frames=atlas_frames,ranked_slots=rank['slots'],
                slot_scores=rank['base_scores'],all_slot_ids=list(range(16))))
            colors={key:pca_color(d[key][:,view],models[f'{key}_{view}']) for key in ['semantics','geometry','fused','post_transformer']}
            def panels(t):
                rgb=d['rgb'][t,view];r=d['routing'][t,view];coords=d['keypoints'][t,view]
                with np.load(study/'turbovla_afa'/f'ep{eid:03d}'/f'f{t:06d}.npz') as a, np.load(study/'turbovla'/f'ep{eid:03d}'/f'f{t:06d}.npz') as b:
                    dense_a=pca_color(a['dino_features'][view],models[f'dino_{view}']).reshape(16,16,3)
                    dense_b=pca_color(b['dino_features'][view],models[f'dino_{view}']).reshape(16,16,3)
                return dict(rgb=rgb,dino_afa=overlay(rgb,d['afa_dino_cls'][t,view]),dino_base=overlay(rgb,d['base_dino_cls'][t,view]),
                    act_afa=overlay(rgb,d['afa_act_map'][t,view]),act_base=overlay(rgb,d['base_act_map'][t,view]),
                    geometry=points(rgb,coords,COLORS,d['attention'][t,view],ellipses=True),routing=colored_routing(rgb,r),
                    semantics=projected_colors(rgb,r,colors['semantics'][t]),
                    keypoints_semantics=points(rgb,coords,colors['semantics'][t]),
                    fused=projected_colors(rgb,r,colors['fused'][t],coords),
                    post_transformer=projected_colors(rgb,r,colors['post_transformer'][t],coords),
                    dino_features_afa=blend(rgb,up(dense_a)),dino_features_base=blend(rgb,up(dense_b)))
            rendered={t:panels(t) for t in selected}
            for t in selected:
                p=rendered[t];frame=f'{ident}_f{t:05d}'
                keys=['rgb','dino_afa','act_afa','geometry','routing','keypoints_semantics','dino_features_afa','semantics','fused']
                board=montage([[p[k] for k in keys[i:i+3]] for i in (0,3,6)])
                emit(board,f'overview/{frame}',f'Episode {eid}, {camera}, frame {t}. Row-major: RGB; AFA DINO CLS; AFA ACT; geometry with one-sigma spatial ellipses; winning competitive query; all keypoints colored by pooled-semantic PCA; dense DINO-feature PCA; pooled-semantic PCA projected by routing; fused-token PCA projected by routing with keypoints. Shared bases/scales across both episodes within camera.')
                single=out/'panels'/ident/f'f{t:05d}';single.mkdir(parents=True,exist_ok=True)
                for key,value in p.items():Image.fromarray(value).save(single/f'{key}.png',dpi=(300,300))
                if t in atlas_frames:
                    tiles=[overlay(p['rgb'],d['attention'][t,view,k]) for k in range(16)]
                    emit(montage([tiles[i:i+4] for i in (0,4,8,12)]),f'all_slots/{frame}_attention',f'All sixteen AFA pooling maps, query IDs 0–15 in row-major order. Episode {eid}, {camera}, frame {t}. Shared 1×–8× log attention lift; each map retains full-grid normalization.')
                    tiles=[masked_rgb(p['rgb'],d['routing'][t,view],k) for k in range(16)]
                    emit(montage([tiles[i:i+4] for i in (0,4,8,12)]),f'all_slots/{frame}_routing_rgb',f'All sixteen competitive-routing RGB responsibility views, IDs 0–15 row-major. Absolute routing probabilities on white. These are input masks, not reconstructions or annotated object segments.')
                    pcent=probability(d['attention'][t,view].reshape(16,256))
                    heatmap(unit(pcent-1/256)@unit(pcent-1/256).T,out/'charts'/f'{frame}_spatial_similarity',vmin=-1,vmax=1,cmap='RdBu_r')
                    heatmap(unit(d['semantics'][t,view])@unit(d['semantics'][t,view]).T,out/'charts'/f'{frame}_semantic_similarity',vmin=-1,vmax=1,cmap='RdBu_r')
            rgbrow=[rendered[t]['rgb'] for t in selected];times=d['timestamp'][selected]
            rows=[rgbrow]+[[rendered[t][key] for t in selected] for key in ['dino_base','dino_afa','act_base','act_afa']]
            emit(montage(rows),f'timelines/{ident}_policies','Rows: RGB; TurboVLA DINO CLS; AFA DINO CLS; TurboVLA ACT; AFA ACT projected through pooling maps. Same twelve frame indices. Cross-checkpoint differences include VLI versus early-BERT.',labels=['RGB','DINO TV','DINO AFA','ACT TV','ACT AFA'],times=times)
            for group in range(4):
                ids=list(range(group*4,group*4+4))
                rows=[rgbrow]+[[overlay(d['rgb'][t,view],d['attention'][t,view,k]) for t in selected] for k in ids]
                emit(montage(rows),f'timelines/{ident}_slots_{group*4:02d}_{group*4+3:02d}',f'Rows: RGB then fixed slot IDs {ids}. Twelve chronological observations; no query reassignment.',labels=['RGB']+[f'S{k:02d}' for k in ids],times=times)
            for name,keys in [('geometry_semantics',['geometry','keypoints_semantics','semantics','fused']),('representation',['dino_features_base','dino_features_afa','semantics','fused','post_transformer'])]:
                emit(montage([rgbrow]+[[rendered[t][key] for t in selected] for key in keys]),f'timelines/{ident}_{name}',f'Rows: RGB; '+ '; '.join(keys)+'. All sixteen slots retained. PCA colors describe continuous embedding variation, not semantic categories.')
            mid=selected[6];rgb=d['rgb'][mid,view];tag=f'{ident}_f{mid:05d}'
            for prefix in ['afa','base']:
                for typ,key in [('dino_layers','dino_layers'),('dino_heads','dino_heads'),('act_queries','act_queries')]:
                    tiles=[overlay(rgb,value) for value in d[f'{prefix}_{key}'][mid,view]]
                    emit(montage([tiles[i:i+4] for i in (0,4,8)]),f'heads_layers/{tag}_{prefix}_{typ}',f'{prefix}, {typ}, indices 0–11 in row-major order. DINO: CLS-to-patch, conditionally normalized after dropping CLS/register keys; ACT: final decoder layer, each of twelve action queries averaged across heads.')
                for layer in range(3):
                    tiles=[overlay(rgb,value) for value in d[f'{prefix}_act_heads'][mid,view,layer]]
                    emit(montage([tiles[:4],tiles[4:]]),f'heads_layers/{tag}_{prefix}_act_layer{layer}_heads',f'{prefix}, ACT layer {layer}, heads 0–7 row-major; averaged over action queries. All weights conditioned on this camera. AFA weights are projected through tokenizer maps.')
            language=d['language'][:,view].mean(1)
            heatmap(language[0],out/'charts'/f'{ident}_query_words',rows=[f'S{k:02d}' for k in range(16)],columns=d['words'],vmax=1,cmap='cividis')
            # Per-head linguistic maps retain absolute probabilities and special tokens.
            for head in range(12):
                heatmap(d['language'][0,view,head],out/'charts'/f'{ident}_query_words_head{head:02d}',columns=d['words'],vmax=1,cmap='cividis')
            heatmap(d['afa_self_attention'][mid,view,0].mean(0),out/'charts'/f'{tag}_slot_self_attention',vmax=1)
            for head in range(4):heatmap(d['afa_self_attention'][mid,view,0,head],out/'charts'/f'{tag}_slot_self_head{head}',vmax=1)
            fig,axes=plt.subplots(1,3,figsize=(11,3.4))
            for ax,key,label in zip(axes,['geometry','semantics','fused'],['Geometry','Pooled semantics','Fused token']):
                values=d[key][:,view];model=models[f'{key}_{view}'];xy=(values-model['mean'])@model['basis'][:,:2]
                for k in range(16):
                    ax.plot(xy[:,k,0],xy[:,k,1],color=COLORS[k],alpha=.4,lw=.5)
                    ax.scatter(xy[selected,k,0],xy[selected,k,1],color=COLORS[k],s=8)
                ax.set(xlabel='PC1',ylabel='PC2',title=label)
                ax.spines[['top','right']].set_visible(False)
            fig.tight_layout();figure_save(fig,out/'charts'/f'{ident}_embedding_paths')
            fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
            p=probability(d['attention'][:,view].reshape(n,16,256))
            entropy=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(256)
            norms=np.linalg.norm(d['geometry'][:,view],axis=-1)/np.maximum(np.linalg.norm(d['semantics'][:,view],axis=-1),1e-9)
            for k in range(16):
                axes[0,0].plot(d['timestamp'],entropy[:,k],color=COLORS[k],lw=.7)
                axes[0,1].plot(d['timestamp'],d['act_weights'][:,view,k],color=COLORS[k],lw=.7)
                axes[0,2].plot(d['timestamp'],norms[:,k],color=COLORS[k],lw=.7)
                xy=d['keypoints'][:,view,k];axes[1,0].plot(d['timestamp'],xy[:,0],color=COLORS[k],lw=.7);axes[1,1].plot(d['timestamp'],xy[:,1],color=COLORS[k],lw=.7)
            axes[1,2].plot(d['timestamp'],d['afa_act_mass'][:,view],label='AFA');axes[1,2].plot(d['timestamp'],d['base_act_mass'][:,view],label='TurboVLA');axes[1,2].legend(frameon=False,fontsize=8)
            for ax,label in zip(axes.flat,['Spatial entropy','ACT slot weight','Geometry / semantic norm','Keypoint x','Keypoint y','ACT camera mass']):
                ax.set_ylabel(label,fontsize=9);ax.spines[['top','right']].set_visible(False)
            for ax in axes[1]:ax.set_xlabel('Time [s]')
            fig.tight_layout();figure_save(fig,out/'charts'/f'{ident}_all_frame_metrics')
            print('Figures',ident,'frames',selected,flush=True)
        fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
        for joint,ax in enumerate(axes.flat):
            ax.plot(d['timestamp'],d['state'][:,joint],label='State',color='#8796a5',lw=.8)
            ax.plot(d['timestamp'],d['recorded_action'][:,joint],label='Recorded command',color='#1e293b',lw=.8)
            ax.plot(d['timestamp'],d['afa_predicted_actions'][:,0,joint],label='AFA replay',color='#dc6635',lw=.8)
            ax.plot(d['timestamp'],d['base_predicted_actions'][:,0,joint],label='TurboVLA replay',color='#297d9e',lw=.8)
            ax.set_ylabel(['Shoulder pan','Shoulder lift','Elbow','Wrist flex','Wrist roll','Gripper'][joint]);ax.spines[['top','right']].set_visible(False)
        axes[0,0].legend(fontsize=6,frameon=False)
        for ax in axes[1]:ax.set_xlabel('Time [s]')
        fig.tight_layout();figure_save(fig,out/'charts'/f'ep{eid:03d}_action_replay')
    return catalog,selections


def browser_assets(out,data,models,episodes):
    """Exact float32 map payloads and original uint8 RGB, loadable via file://."""
    root=out/'viewer';root.mkdir(exist_ok=True)
    for eid,n in episodes:
        d={key[len(f'ep{eid}_'):]:v for key,v in data.items() if key.startswith(f'ep{eid}_')}
        folder=root/'data'/f'ep{eid:03d}';folder.mkdir(parents=True,exist_ok=True)
        for t in range(n):
            arrays={key:d[key][t] for key in ['attention','routing','keypoints']}
            for key in ['dino_cls','act_map']:
                for prefix in ['afa','base']:arrays[f'{prefix}_{key}']=d[f'{prefix}_{key}'][t]
            for key in ['semantics','fused','post_transformer']:
                arrays[f'{key}_colors']=np.stack([pca_color(d[key][t,v],models[f'{key}_{v}']) for v in (0,1)])
            payload={key:base64.b64encode(np.asarray(value,dtype='<f4').tobytes()).decode('ascii') for key,value in arrays.items()}
            record=dict(episode=eid,frame=t,time=float(d['timestamp'][t]),arrays=payload,
                act_mass=d['act_mass'][t].tolist(),base_act_mass=d['base_act_mass'][t].tolist())
            record['rgb']=[]
            for v in (0,1):
                buffer=io.BytesIO();Image.fromarray(d['rgb'][t,v]).save(buffer,format='PNG')
                record['rgb'].append('data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'))
            (folder/f'f{t:06d}.js').write_text('window.loadDiagnosticFrame('+json.dumps(record,separators=(',',':'))+');\n')
    lut=(plt.get_cmap('magma')(np.linspace(0,1,256))[:,:3]*255).round().astype(int).tolist()
    template=Path(__file__).with_name('successgrasp_viewer.html').read_text()
    template=template.replace('__EPISODES__',json.dumps(dict(episodes))).replace('__COLORS__',json.dumps((COLORS*255).round().astype(int).tolist())).replace('__MAGMA__',json.dumps(lut))
    (root/'index.html').write_text(template)


def documents(study,out,catalog,selections,report,episodes):
    source=json.loads((study/'source.json').read_text())
    (out/'selection.json').write_text(json.dumps(dict(selection=selections,all_frames_analyzed=True,
        total_frames_per_policy=sum(n for _,n in episodes),temporal_frames_per_sequence=12,all_slot_ids=list(range(16)),
        selection_method='Twelve equal temporal bins, highest .65 slot quality + .35 RGB sharpness per bin; four top slots ranked for reference, all sixteen rendered',
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    report.update(figure_pairs=len(catalog)+len(list((out/'charts').glob('*.png'))),clean_image_pairs=len(catalog),
        charts=len(list((out/'charts').glob('*.png'))),viewer_frames=sum(n for _,n in episodes),viewer_views=2*sum(n for _,n in episodes))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    chart_records=[]
    for path in sorted((out/'charts').glob('*.png')):
        name=path.stem
        if 'query_words' in name:
            caption='Early-BERT query-to-word attention, all sixteen query rows and all BERT tokens including special tokens. Absolute probability scale 0–1. The filename identifies an individual head, or the average across twelve heads. These weights are temporally invariant for this fixed instruction and camera.'
        elif 'slot_self' in name:
            caption='AFA post-tokenizer self-attention, query versus key slot IDs 0–15. Absolute probabilities 0–1. The filename identifies a head or the average over four heads. This is token mixing, not a segmentation mask.'
        elif 'spatial_similarity' in name:
            caption='Centered cosine similarity of all sixteen spatial pooling distributions, after subtracting the uniform floor. Shared scale −1 to 1. Off-diagonal similarity indicates spatial redundancy.'
        elif 'semantic_similarity' in name:
            caption='Cosine similarity of all sixteen projected pooled semantic embeddings. Shared scale −1 to 1. This describes feature similarity rather than spatial overlap or annotated object categories.'
        elif 'embedding_paths' in name:
            caption='Separate PCA trajectories for the geometric component, pooled semantic component and fused tokens. Colors denote fixed query IDs; paths cover every frame and dots mark the twelve displayed observations. Components are fitted across both episodes within camera; axes from different representation spaces are not interchangeable.'
        elif 'all_frame_metrics' in name:
            caption='All-frame diagnostics for all sixteen queries. Row-major: normalized spatial entropy; ACT slot weight; geometric/semantic component norm ratio; keypoint x; keypoint y; absolute ACT mass assigned to this camera in the two policies. Fixed query colors. Coordinates refer to the full normalized model grid.'
        else:
            caption='Offline action replay across all frames. Six robot dimensions: recorded state, recorded executed commands, and first action in the AFA and TurboVLA predicted chunks. Predictions are not executed; this is descriptive command agreement on a fixed recorded observation stream, not closed-loop task performance.'
        chart_records.append(dict(name='charts/'+name,caption=caption,png='charts/'+path.name,pdf='charts/'+path.with_suffix('.pdf').name))
    all_records=catalog+chart_records
    (out/'figure_manifest.json').write_text(json.dumps(all_records,indent=2))
    (out/'CAPTIONS.md').write_text('# Figure captions\n\n'+ '\n\n'.join(f"## {r['name']}\n\n{r['caption']}" for r in all_records))
    lines=['# Success-grasp: diagnostica completa','', '[Metodo e significato](README.md) · [Didascalie](CAPTIONS.md) · [Tutte le figure](clean/) · [Grafici diagnostici](charts/) · [Viewer da scaricare](viewer/index.html)','',
        'Il viewer va aperto localmente dopo il download: GitHub mostra il codice HTML. Include ogni frame di entrambi gli episodi e consente di cambiare camera, query e tipo di visualizzazione.','', '## Sequenze temporali','']
    for s in selections:
        ident=f"ep{s['episode']:03d}_{s['camera']}"
        lines += [f'### {ident}','',f"Frame visualizzati: {s['frames']}. Tutti i 16 slot sono conservati nelle quattro tavole temporali.",'',
            f'[Confronto policy PDF](clean/timelines/{ident}_policies.pdf) · [Geometria e semantica PDF](clean/timelines/{ident}_geometry_semantics.pdf) · [Rappresentazioni PDF](clean/timelines/{ident}_representation.pdf)','',
            f'![Geometria e semantica](clean/timelines/{ident}_geometry_semantics.png)','',
            f'![Confronto](clean/timelines/{ident}_policies.png)','']
        for group in range(4):
            name=f'{ident}_slots_{group*4:02d}_{group*4+3:02d}'
            lines += [f'<details><summary>Slot {group*4}–{group*4+3}</summary>','',f'[PDF](clean/timelines/{name}.pdf)','',f'![Slot](clean/timelines/{name}.png)','','</details>','']
    lines += ['## Tutti i frame selezionati: overview e tutti gli slot','']
    for r in catalog:
        if r['name'].startswith('overview/'):
            name=r['name'].split('/')[-1];tag=name
            lines += [f'<details><summary>{tag}</summary>','',f"[PDF]({r['pdf']}) · [Pannelli separati](panels/{tag.rsplit('_f',1)[0]}/f{tag.rsplit('_f',1)[1]})",'',f"![Overview]({r['png']})",'']
            for kind in ['attention','routing_rgb']:
                p=f'clean/all_slots/{name}_{kind}'
                if (out/f'{p}.png').exists():lines += [f'[PDF {kind}]({p}.pdf)','',f'![{kind}]({p}.png)','']
            lines += ['</details>','']
    lines += ['## Head, layer e query delle azioni','']
    for r in catalog:
        if r['name'].startswith('heads_layers/'):
            lines += [f"<details><summary>{r['name']}</summary>",'',f"[PDF]({r['pdf']})",'',f"![Attention]({r['png']})",'','</details>','']
    lines += ['## Query linguistiche, interazioni e metriche','']
    for p in sorted((out/'charts').glob('*.png')):
        name=p.stem
        lines += [f'<details><summary>{name}</summary>','',f'[PDF](charts/{name}.pdf)','',f'![{name}](charts/{p.name})','','</details>','']
    (out/'GALLERY.md').write_text('\n'.join(lines))
    readme=f'''# Diagnostica integrale del rollout successgrasp

[Galleria completa](GALLERY.md) · [Didascalie](CAPTIONS.md) · [Frame e query](selection.json) · [Verifiche](validation.json)

## Dati e copertura

Dataset: [{source['repo_id']}](https://huggingface.co/datasets/{source['repo_id']}), revisione `{source['revision']}`. Contiene due episodi del task “Grab the peg and push the red cube”: episodio 0, 349 frame; episodio 1, 357 frame, a 30 Hz e con due camere. **Tutti i 706 frame sono elaborati da entrambi i checkpoint**, per 1.412 predizioni di chunk e 2.824 viste. Il nome successgrasp identifica il materiale indicato dall'utente: la riuscita dell'intero task di pushing non è verificata automaticamente e non viene inferita dalle attention map. Il repository non certifica quale checkpoint registrò originariamente i video; i checkpoint qui indicati sono quelli usati nel replay.

Si usano lo stesso AFA early-BERT finale e lo stesso TurboVLA VLI fine-tuned delle analisi precedenti, entrambi a 160000 micro-step. I percorsi e gli hash sono nei manifest in `provenance/`. Inferenza FP32, batch uno, evaluation/inference mode, senza AMP o TF32. Nessun training, nessuna azione eseguita sul robot.

Le immagini, gli stati, le azioni registrate e i timestamp sono verificati identici fra i due replay. Per **ogni frame** si confronta la predizione con registrazione diagnostica attiva e disattiva. Le verifiche risultanti sono in `validation.json` e nei due report di copertura.

## Tensori estratti su ogni frame

- DINO: feature dense 256×768 per camera; CLS→key per tutti i 12 layer e tutte le 12 head, inclusi i pesi su CLS e quattro register token. Le figure spaziali eliminano questi cinque prefissi e normalizzano sulla griglia di 256 patch. Si conservano anche le masse prima della normalizzazione.
- AFA: tutte le 16 mappe di pooling per camera, routing competitivo P(query|patch), coordinate geometriche, valori aggregati 768D, proiezione semantica 256D, proiezione geometrica 256D e token fusi 256D. La relazione **fused = geometry + semantics** viene verificata sui dati.
- Query: vettori appresi prima e dopo il condizionamento early-BERT, attenzione query→parola su tutte le 12 head, padding e token speciali conservati. A parità di istruzione e camera questi pesi possono restare invarianti nel tempo: ciò non dimostra un'attenzione linguistica che segue l'immagine.
- Transformer AFA: token dopo il mixing e matrici self-attention delle quattro head, ottenute con replay separato. Non si modificano i moduli interni del forward produttivo. Si verifica l'accordo fra replay e uscita reale del transformer.
- ACT: cross-attention di tutti e tre i layer, tutte le otto head e tutte le dodici query di azione; chunk denormalizzati 12×6 e azioni/stati registrati.

Per TurboVLA le mappe ACT sono pesi sulle patch dense. Per AFA sono proiezioni dei pesi degli slot attraverso le mappe di pooling. Si condiziona alla camera per la distribuzione spaziale; le masse assolute per camera sono salvate e rappresentate separatamente. Questa proiezione non annulla il successivo mixing dei token e non è una salienza causale o una attention rollout completa. I due checkpoint differiscono anche nell'interazione linguistica VLI/early-BERT: il confronto non isola causalmente il tokenizer.

Il calcolo CLS ottimizzato a singola riga è verificato contro eager. Su una head molto concentrata del primo layer TurboVLA la forma ridotta della GEMM dava uno scarto numerico apprezzabile: per il replay della baseline si usa perciò `analyze_successgrasp_fullqk.py`, che conserva la forma QK completa di eager, senza cambiare SDPA nel forward reale. Entrambi gli script e i loro hash sono conservati. Le formule delle mappe sono le stesse.

## Significato delle figure

| Famiglia | Cosa rappresenta |
|---|---|
| `overview` | Nove pannelli: RGB, DINO CLS AFA, ACT AFA, geometria ed ellissi, routing vincente, keypoint colorati dalla semantica, feature DINO in PCA, semantica aggregata proiettata sull'RGB, token fusi proiettati con keypoint. |
| `all_slots/*attention` | Tutte le mappe spaziali dei 16 slot, ID 0–15 per righe. Mostrano quali patch ciascuna query aggrega. |
| `all_slots/*routing_rgb` | Tutte le responsabilità P(query|patch), applicate all'RGB su fondo bianco. Non sono ricostruzioni o segmentazioni annotate. |
| `timelines/*slots*` | Gli stessi ID lungo dodici osservazioni. Quattro tavole coprono tutti i 16 slot; nessun riordinamento nel tempo. |
| `timelines/*policies` | RGB; DINO TurboVLA; DINO AFA; ACT TurboVLA; ACT AFA. Gli stessi frame in tutte le righe. |
| `timelines/*geometry_semantics` | RGB; geometria e dispersione; keypoint colorati dalla semantica aggregata; proiezione della semantica; proiezione del token geometrico+semantico. |
| `timelines/*representation` | RGB; feature DINO TurboVLA; feature DINO AFA; semantica AFA; token fuso; token dopo il transformer. Ogni spazio ha una PCA esplicitamente distinta. |
| `heads_layers` | Layer DINO, head DINO finali, head ACT per ciascun layer e dodici query di azione. L'ordine è per righe, a partire da zero. |
| `query_words` | Pesi reali query→token BERT, mediati sulle head o separati per ciascuna head; tutti i 16 slot e token speciali. Scala assoluta 0–1. |
| `slot_self_attention` | Interazioni fra i 16 token nel transformer AFA, media o singole head. Pesi 0–1. |
| `spatial_similarity` / `semantic_similarity` | Coseno fra mappe spaziali dopo sottrazione dell'uniforme, oppure fra embedding semantici. Ridondanza spaziale e somiglianza semantica sono proprietà distinte. |
| `embedding_paths` | Traiettorie PCA di geometria, semantica e token fuso: colori degli ID fissi, punti ai frame mostrati. Non dimostrano cluster di oggetti. |
| `all_frame_metrics` | Entropia, peso ACT di ciascuno slot, rapporto delle norme geometria/semantica, coordinate e massa ACT per camera, su tutti i frame. |
| `action_replay` | Stato, comandi registrati e prima azione del chunk ricalcolato dalle due policy. Il replay è open-loop sulle osservazioni registrate: la differenza dai comandi non è una misura del successo di una nuova esecuzione. |

Le ellissi descrivono una deviazione standard delle distribuzioni spaziali; non sono extent predetti dal modello. I keypoint usano la vera griglia dei centri delle patch. Quelli nel padding possono scomparire nel crop grafico: non vengono spostati artificialmente. Mappe numericamente inattive e slot ridondanti restano inclusi.

## PCA e colori semantici

Per ciascuna camera si adatta una PCA separata per geometria, semantica proiettata, token fuso e token dopo transformer, usando **tutti gli slot di tutti i 706 frame**, con base e limiti condivisi fra episodi. Le tre componenti principali diventano canali RGB, con segni fissati e limiti ai quantili 2–98%. Questi colori rappresentano variazioni continue degli embedding e non nomi di classi o cluster annotati. Colori di spazi PCA diversi non sono direttamente confrontabili.

Per le feature DINO si usa una base comune ai due checkpoint entro ciascuna camera, calibrata deterministicamente sui frame 0,16,32,… e sull'ultimo frame di ogni episodio, escludendo le righe di padding dalla calibrazione. Questa è una riduzione dimensionale descrittiva in coordinate condivise; non garantisce allineamento semantico dei due encoder. I modelli PCA e la varianza spiegata sono in `pca_models.npz`.

Le proiezioni semantiche sull'RGB combinano il colore di tutti i token con il routing competitivo originale. Il colore sul keypoint invece appartiene alla semantica dello slot in quella posizione. Non si assegnano etichette manuali alle regioni.

## Selezione e stile

L'inferenza riguarda tutti i frame. Solo dopo si scelgono dodici osservazioni cronologiche per camera, una per ciascuno di dodici intervalli temporali uniformi, favorendo nitidezza e leggibilità con il criterio già documentato in [full_episodes](../full_episodes/README.md). Sei di questi frame ricevono anche atlanti di tutti gli slot e maschere RGB; un frame centrale riceve gli atlanti per head/layer/query. Il viewer permette di esaminare i frame intermedi e tutte le query senza questa selezione. Gli indici esatti sono in `selection.json`. Le figure restano illustrative e la selezione post-hoc non è una stima imparziale di superiorità del tokenizer.

Le immagini `clean/` non contengono titoli o testo. Il padding superiore e inferiore di 32 pixel viene rimosso **solo nel rendering**, dopo l'inferenza e l'interpolazione, lasciando celle 256×192. Non si tagliano aree scure della scena e non si rinormalizzano le probabilità dopo il crop. Le attention map condividono la scala log₂ rispetto all'uniforme, da 1× a 8×, e l'opacità massima 0.65. Le maschere RGB usano probabilità assolute. I grafici numerici mantengono gli assi necessari; le versioni `minimal_labels/` aggiungono soltanto nomi delle righe e timestamp. PNG e PDF mantengono la risoluzione nativa delle celle, senza fingere dettaglio spaziale oltre la griglia 16×16.

## Accesso e riproducibilità

Aprire `viewer/index.html` **localmente dopo aver scaricato la raccolta**: non richiede server, training, rete o librerie esterne. Il viewer carica i dati float32 delle mappe e PNG RGB originali di ogni frame; permette di scegliere episodio, camera, query e modalità e scaricare il canvas senza testo. Il viewer è uno strumento di esplorazione con interpolazione del browser; i file PNG/PDF pre-esportati usano il renderer scientifico Python. `panels/` contiene anche ogni pannello separato dei frame selezionati.

L'archivio completo di tutti i tensori per ogni frame rimane in `{study}`. GitHub contiene le mappe di tutti i frame necessarie al viewer, figure, PCA, metriche e un sottoinsieme degli archivi completi in `selected_arrays/`. I file per frame completi di entrambi i checkpoint non sono tutti duplicati su GitHub. Gli script sono in `scripts/` nella repository principale; richiedono l'ambiente LeRobot e i pacchetti locali delle policy per ripetere l'estrazione. Il rendering legge gli archivi salvati senza eseguire inferenza.

`slot_metrics.csv` contiene una riga per ogni frame/camera/slot; `frame_metrics.csv` riporta le statistiche per camera. I manifest fissano checkpoint e revisione del dataset. `figure_manifest.json` e `CAPTIONS.md` descrivono l'ordine dei pannelli.
'''
    (out/'README.md').write_text(readme)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.study/'turbovla_afa/manifest.json').read_text())
    episodes=[(ep['episode_index'],ep['length']) for ep in manifest['episodes']]
    for kind in ['turbovla_afa','turbovla']:
        coverage=json.loads((args.study/kind/'coverage.json').read_text())
        assert coverage['completed_frames']==coverage['expected_frames']==sum(n for _,n in episodes)
        assert coverage['hook_checks']==coverage['expected_frames']
    data=summarize(args.study,args.output)
    report=metric_tables(data,args.output,episodes)
    models=make_models(data,episodes,args.output)
    catalog,selections=export_figures(args.study,args.output,data,models,episodes)
    browser_assets(args.output,data,models,episodes)
    provenance=args.output/'provenance';provenance.mkdir(exist_ok=True)
    shutil.copy2(args.study/'source.json',provenance/'source.json')
    for kind in ['turbovla_afa','turbovla']:
        for name in ['manifest.json','coverage.json','fullqk_provenance.json']:
            path=args.study/kind/name
            if path.exists():shutil.copy2(path,provenance/f'{kind}_{name}')
    selected=args.output/'selected_arrays';selected.mkdir(exist_ok=True)
    for s in selections:
        eid=s['episode'];frame=s['frames'][6]
        for kind in ['turbovla_afa','turbovla']:
            src=args.study/kind/f'ep{eid:03d}'/f'f{frame:06d}.npz'
            shutil.copy2(src,selected/f'{kind}_ep{eid:03d}_f{frame:06d}.npz')
    documents(args.study,args.output,catalog,selections,report,episodes)
    print('Complete',json.dumps(report),flush=True)


if __name__=='__main__':main()
