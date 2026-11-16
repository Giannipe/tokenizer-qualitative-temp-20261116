#!/usr/bin/env python
"""Rank every frame/slot; export padding-free strips with captions kept separately."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def normalized(x):
    lo,hi=np.quantile(x,[.1,.9])
    return np.clip((x-lo)/max(hi-lo,1e-8),0,1)


def rank_episode(d, view):
    raw=d['attention'][:,view].reshape(len(d['frame_index']),16,256).astype(np.float64)
    p=raw/np.maximum(raw.sum(-1,keepdims=True),1e-12)
    entropy=-(p*np.log(np.maximum(p,1e-12))).sum(-1)/np.log(256)
    visible=p.reshape(-1,16,16,16)[:,:,2:14].sum((-1,-2))
    alive=np.clip(raw.sum(-1),0,1)
    variation=np.zeros_like(entropy)
    variation[1:]=.5*np.abs(np.diff(p,axis=0)).sum(-1)
    coherence=1-variation
    relevance=d['act_weights'][:,view]/np.maximum(d['act_mass'][:,view,None],1e-12)
    relevance=np.clip(relevance*16/2,0,1)
    quality=(.42*(1-entropy)+.22*visible+.16*coherence+.20*relevance)*alive
    base=quality.mean(0)
    centered=p-1/256
    centered/=np.maximum(np.linalg.norm(centered,axis=-1,keepdims=True),1e-8)
    similarity=np.einsum('tkn,tln->kl',centered,centered)/len(p)
    chosen=[]
    for _ in range(4):
        score=base.copy()
        if chosen:
            score-=.35*np.maximum(similarity[:,chosen].max(-1),0)
        score[chosen]=-np.inf
        chosen.append(int(score.argmax()))
    rgb=d['rgb'][:,view].astype(np.float32)/255
    gray=rgb.mean(-1)
    lap=gray[:,2:,1:-1]+gray[:,:-2,1:-1]+gray[:,1:-1,2:]+gray[:,1:-1,:-2]-4*gray[:,1:-1,1:-1]
    sharpness=normalized(np.abs(lap).mean((1,2)))
    frame_quality=.65*quality[:,chosen].mean(-1)+.35*sharpness
    state=d['state'].astype(float)
    scale=np.maximum(state.std(0),1)
    motion=np.r_[0,np.linalg.norm(np.diff(state,axis=0)/scale,axis=-1)]
    cumulative=np.cumsum(motion)
    movement=cumulative/cumulative[-1] if cumulative[-1]>1e-8 else np.linspace(0,1,len(p))
    progress=.5*movement+.5*np.linspace(0,1,len(p))
    frames=[]
    for i in range(8):
        candidates=np.flatnonzero((progress>=i/8)&(progress<(i+1)/8 if i<7 else progress<=1))
        gap=max(2,len(p)//50)
        lower=frames[-1]+gap if frames else 0
        upper=len(p)-1-gap*(7-i)
        candidates=candidates[(candidates>=lower)&(candidates<=upper)]
        if not len(candidates):
            center=int(np.clip(round((i+.5)*len(p)/8),lower,upper))
            candidates=np.arange(max(lower,center-len(p)//24),min(upper,center+len(p)//24)+1)
        score=frame_quality[candidates]-.15*np.abs(progress[candidates]-(i+.5)/8)
        frames.append(int(candidates[score.argmax()]))
    frames=list(dict.fromkeys(frames))
    if len(frames)!=8 or frames!=sorted(frames):
        raise ValueError('Frame selection must contain eight chronological observations')
    return dict(slots=chosen,frames=frames,base_scores=base.tolist(),similarity=similarity.tolist(),
        entropy=entropy,visible=visible,alive=alive,quality=quality,frame_quality=frame_quality,
        sharpness=sharpness,progress=progress,selection_score=float(frame_quality[frames].mean()))


def overlay(rgb, distribution):
    p=np.asarray(distribution,dtype=float)
    p=p/max(p.sum(),1e-12)
    lift=np.clip(np.log2(np.maximum(p*256,1)),0,3)/3
    full=np.asarray(Image.fromarray(lift.astype(np.float32)).resize((256,256),Image.Resampling.BILINEAR))
    strength=full[32:224]
    colors=plt.get_cmap('magma')(strength)[...,:3]
    alpha=.65*strength[...,None]
    return np.round(np.clip(rgb/255*(1-alpha)+colors*alpha,0,1)*255).astype(np.uint8)


def montage(rows):
    gutter=3
    height=192*len(rows)+gutter*(len(rows)-1)
    width=256*len(rows[0])+gutter*(len(rows[0])-1)
    result=np.full((height,width,3),255,dtype=np.uint8)
    for r,row in enumerate(rows):
        for c,tile in enumerate(row):
            if tile.shape!=(192,256,3):raise ValueError(f'Unexpected cropped tile shape {tile.shape}')
            result[r*(192+gutter):r*(192+gutter)+192,c*(256+gutter):c*(256+gutter)+256]=tile
    return Image.fromarray(result)


def with_labels(clean, labels, timestamps):
    image=Image.new('RGB',(clean.width+105,clean.height+31),'white')
    image.paste(clean,(105,31))
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),19)
    for r,label in enumerate(labels):draw.text((5,31+r*195+84),label,font=font,fill='#263241')
    for c,t in enumerate(timestamps):draw.text((105+c*259+80,3),f'{t:.2f}s',font=font,fill='#263241')
    return image


def save_pair(image,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path.with_suffix('.png'),dpi=(300,300))
    fig=plt.figure(figsize=(image.width/300,image.height/300),dpi=300)
    ax=fig.add_axes([0,0,1,1]);ax.imshow(image,interpolation='none');ax.axis('off')
    fig.savefig(path.with_suffix('.pdf'),dpi=300,bbox_inches=None,pad_inches=0)
    plt.close(fig)


def export(study, destination):
    afa=study/'turbovla_afa';base=study/'turbovla'
    am=json.loads((afa/'manifest.json').read_text());bm=json.loads((base/'manifest.json').read_text())
    assert am['episodes']==bm['episodes']
    source={f"{r['group']}_task{r['task_id']}_ep{r['episode']:03d}":r for r in am['episodes']}
    destination.mkdir(parents=True,exist_ok=True)
    records=[];coverage=[];all_metrics=[]
    selected_dir=destination/'selected_arrays';selected_dir.mkdir(exist_ok=True)
    for name,r in source.items():
        path=afa/f'{name}.npz';other=base/f'{name}.npz'
        with np.load(path) as archive:d={k:archive[k] for k in archive.files}
        with np.load(other) as archive:b={k:archive[k] for k in archive.files}
        expected=r['episode_metadata']['length']
        np.testing.assert_array_equal(d['frame_index'],np.arange(expected))
        np.testing.assert_array_equal(d['frame_index'],b['frame_index'])
        np.testing.assert_array_equal(d['state'],b['state'])
        np.testing.assert_array_equal(d['timestamp'],b['timestamp'])
        np.testing.assert_array_equal(d['video_sync_error_s'],b['video_sync_error_s'])
        for data in (d,b):
            for key,value in data.items():
                if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError((name,key))
        coverage.append(dict(episode=name,expected_frames=expected,afa_frames=len(d['frame_index']),
            turbovla_frames=len(b['frame_index']),first_frame=0,last_frame=expected-1,
            afa_archive_sha256=sha256_file(path),turbovla_archive_sha256=sha256_file(other)))
        for view in (0,1):
            selection=rank_episode(d,view)
            indices=selection['frames'];slots=selection['slots']
            for t in range(expected):
                for k in range(16):
                    all_metrics.append(dict(episode=name,group=r['group'],task=r['task_id'],camera=view,
                        frame=t,slot=k,entropy=float(selection['entropy'][t,k]),visible_mass=float(selection['visible'][t,k]),
                        alive_mass=float(selection['alive'][t,k]),quality=float(selection['quality'][t,k]),
                        selected_slot=k in slots,selected_frame=t in indices))
            images=[d['rgb'][t,view] for t in indices]
            rows=[images,[overlay(rgb,d['dino_cls'][t,view]) for rgb,t in zip(images,indices)]]
            for k in slots:rows.append([overlay(rgb,d['attention'][t,view,k]) for rgb,t in zip(images,indices)])
            clean=montage(rows)
            ident=f'{name}_{["front","wrist"][view]}'
            token_path=destination/'clean'/f'{ident}_tokens'
            save_pair(clean,token_path)
            save_pair(with_labels(clean,['RGB','DINO']+[f'S{k:02d}' for k in slots],d['timestamp'][indices]),
                destination/'minimal_labels'/f'{ident}_tokens')
            rows=[images]
            for data,key in [(b,'dino_cls'),(d,'dino_cls'),(b,'act_map'),(d,'act_map')]:
                rows.append([overlay(rgb,data[key][t,view]) for rgb,t in zip(images,indices)])
            comparison=montage(rows)
            save_pair(comparison,destination/'clean'/f'{ident}_policies')
            save_pair(with_labels(comparison,['RGB','DINO TV','DINO AFA','ACT TV','ACT AFA'],d['timestamp'][indices]),
                destination/'minimal_labels'/f'{ident}_policies')
            compact={key:value[indices] for key,value in d.items()}
            for key in ['dino_cls','dino_patch_mass','act_map','act_mass','act_weights','predicted_actions']:
                compact[f'turbovla_{key}']=b[key][indices]
            np.savez_compressed(selected_dir/f'{ident}.npz',**compact)
            entry=dict(id=ident,group=r['group'],task_id=r['task_id'],episode=r['episode'],view=view,
                slots=slots,frames=indices,times=d['timestamp'][indices].tolist(),
                slot_base_scores=selection['base_scores'],selection_score=selection['selection_score'],
                tokens=f'clean/{ident}_tokens',policies=f'clean/{ident}_policies')
            records.append(entry)
            print('Exported',ident,'slots',slots,'frames',indices,flush=True)
    with (destination/'all_frame_slot_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_metrics)
    recommended=[]
    for task in (1,2,3):
        candidates=[r for r in records if r['task_id']==task and r['group']=='verified_demo']
        recommended.append(max(candidates,key=lambda r:r['selection_score'])['id'])
    selection_doc=dict(episodes=coverage,selection=records,recommended=recommended,
        total_frames_per_policy=sum(x['expected_frames'] for x in coverage),
        crop_xyxy=[0,32,256,224],crop_for_display_only=True,probability_reference_patches=256,
        frames_per_strip=8,slot_ids_fixed_within_strip=True,selection_is_post_hoc_illustrative=True,
        selector_sha256=sha256_file(Path(__file__)))
    (destination/'selection.json').write_text(json.dumps(selection_doc,indent=2))
    write_docs(destination,records,recommended)
    with zipfile.ZipFile(destination/'selected_clean_figures.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for ident in recommended:
            for kind in ['tokens','policies']:
                for extension in ['png','pdf']:
                    p=destination/'clean'/f'{ident}_{kind}.{extension}';archive.write(p,p.relative_to(destination))
        for p in ['selection.json','CAPTIONS.md']:archive.write(destination/p,p)


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write_docs(root,records,recommended):
    captions=['# Didascalie e ordine delle righe','',
        'Le immagini clean non contengono testo, bordi neri di preprocessing o titoli. Le versioni minimal_labels aggiungono solo nomi delle righe e timestamp.',
        '', 'La prima riga è sempre RGB. Nei pannelli tokens seguono DINO CLS→patch del checkpoint AFA e le quattro query elencate. Nei pannelli policies seguono DINO CLS TurboVLA, DINO CLS AFA, ACT TurboVLA, ACT AFA proiettata tramite le mappe del tokenizer.',
        '', 'Le scale sono condivise: log2 dell’attenzione rispetto all’uniforme su 256 patch, saturato a 8×. I valori sotto uniforme sono trasparenti. Si elimina solo il padding nel display dopo aver calcolato e interpolato le mappe; non si rinormalizza sulla parte ritagliata.',
        '', 'La selezione è post-hoc e illustrativa: usa tutte le osservazioni, favorisce leggibilità e differenziazione delle query AFA, e non misura un vantaggio statistico o causale rispetto alla baseline. Gli stessi frame vengono usati per entrambi i checkpoint. Non si cambiano query fra colonne per simulare un tracking migliore.', '']
    gallery=['# Analisi completa: immagini pulite','', '[Metodo e copertura](README.md) · [Didascalie](CAPTIONS.md) · [Scarica le figure consigliate](selected_clean_figures.zip)','', '## Selezione consigliata','']
    for r in records:
        text=(f"Task {r['task_id']}, {r['group']}, episodio {r['episode']}, camera {['frontale','polso'][r['view']]}. "
              f"Frame: {r['frames']}. Query AFA, dall’alto al basso: {r['slots']}. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.")
        captions += [f"## {r['id']}",'',text,'']
        if r['id'] in recommended:
            gallery += [f"### Task {r['task_id']} — {r['id']}",'',
                f"[PDF token]({r['tokens']}.pdf) · [PNG token]({r['tokens']}.png) · [Confronto policy]({r['policies']}.pdf) · [Etichette minime](minimal_labels/{r['id']}_tokens.png)",'',
                f"![{r['id']}]({r['tokens']}.png)",'']
    gallery += ['## Tutte le sequenze','']
    for r in records:
        gallery += [f"<details><summary>{r['id']} — slot {r['slots']}</summary>",'',
            f"[Token PDF]({r['tokens']}.pdf) · [Policy PDF]({r['policies']}.pdf)",'',
            f"![Token]({r['tokens']}.png)",'',f"![Policy]({r['policies']}.png)",'','</details>','']
    (root/'GALLERY.md').write_text('\n'.join(gallery))
    (root/'CAPTIONS.md').write_text('\n'.join(captions))
    cards=''.join(f'<article><h2>{r["id"]}</h2><a href="{r["tokens"]}.pdf">PDF token</a> · <a href="{r["policies"]}.pdf">PDF policy</a><img loading="lazy" src="{r["tokens"]}.png"><img loading="lazy" src="{r["policies"]}.png"></article>' for r in sorted(records,key=lambda r:r['id'] not in recommended))
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Full episode qualitative study</title><style>body{max-width:1600px;margin:32px auto;font:16px system-ui}img{width:100%}article{margin:48px 0}</style><h1>Full-episode study · clean figures</h1>'+cards)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    export(a.study,a.output)


if __name__=='__main__':main()
