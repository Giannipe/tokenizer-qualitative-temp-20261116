#!/usr/bin/env python
"""Separate a normalization intervention from a confounded trained-model comparison."""
import argparse,json,types
from pathlib import Path
import numpy as np
from extract_paper_attention import raw_batch
from tokenizer_qualitative import sha256

COMP=Path('/data2/gpepe/lerobot_runs/turbovla_afa_early_bert_multitask_full_80k_20261110/checkpoints/160000/pretrained_model')
NOCOMP=Path('/data2/gpepe/lerobot_runs/turbovla_afa_mh12_vli_no_competitive_multitask_full_80k_20261116/checkpoints/160000/pretrained_model')

def extract(out):
    import torch,lerobot_policy_turbovla_afa
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class,make_pre_post_processors
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=out.parent;records=[]
    for task,folder,study in [(1,'cube_verified','/data2/gpepe/qualitative_verified_complete_20261117/cube'),(2,'plush_verified','/data2/gpepe/qualitative_verified_complete_20261117/plush'),(3,'successgrasp','/data2/gpepe/qualitative_successgrasp_20261116')]:
        study=Path(study);source=json.loads((study/'source.json').read_text());meta=json.loads((study/'turbovla_afa/manifest.json').read_text())
        ep=meta['episodes'][0];eid=ep['episode_index']
        sequence=next(s for s in json.loads((parent/folder/'selection.json').read_text())['selection'] if s['episode']==eid and s['camera']=='front')
        for position in [3,8]:
            t=sequence['frames'][position]
            records.append(dict(id=f'task{task}_ep{eid:03d}_f{t:05d}',root=source['local_root'],task=ep['tasks'][0],task_id=task,
                episode=eid,frame=t,episode_metadata=ep,fps=30,source=source))
    report=dict(records=records,checkpoints={},hook_max_error=0.,controlled_routing_max_error=0.,controlled_pooling_max_error=0.)
    (out/'arrays').mkdir(exist_ok=True,parents=True)
    for label,checkpoint in [('competitive',COMP),('trained_no_competitive',NOCOMP)]:
        config=PreTrainedConfig.from_pretrained(checkpoint,cli_overrides=['--device=cpu','--inference_dtype=null','--use_amp=false','--vision.compute_precision=fp32','--interaction.compute_precision=fp32'])
        policy=get_policy_class('turbovla_afa').from_pretrained(checkpoint,config=config,local_files_only=True,strict=True).eval().to('cuda:0')
        pre,post=make_pre_post_processors(config,pretrained_path=checkpoint,preprocessor_overrides={'device_processor':{'device':'cpu'}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
        report['checkpoints'][label]=dict(path=str(checkpoint),sha256=sha256(checkpoint/'model.safetensors'),config=json.loads((checkpoint/'config.json').read_text()))
        captured={};active=[True];handles=[];originals=[]
        for view,pool in enumerate(policy.model.afa_tokenizers):
            pool.record_inference_diagnostics=True
            def hook(module,inputs,output,v=view):
                if active[0]:
                    captured[f'attention_{v}']=module.latest_attention[0].numpy()
                    if module.latest_query_softmax is not None:captured[f'routing_{v}']=module.latest_query_softmax[0].numpy()
            handles.append(pool.register_forward_hook(hook))
            original=pool._project_cross_attention;originals.append((pool,original))
            def project(self,*inputs,v=view,fn=original):
                result=fn(*inputs)
                if active[0] and label=='competitive':captured[f'logits_{v}']=result[0][0].detach().cpu().numpy()
                return result
            pool._project_cross_attention=types.MethodType(project,pool)
        tables={}
        with torch.inference_mode():
            for r in records:
                raw,pts=raw_batch(r,config,tables);cpu=pre(raw)
                batch={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
                captured.clear();active[0]=True;pred=policy.predict_action_chunk(batch)
                active[0]=False;baseline=policy.predict_action_chunk(batch)
                error=float((pred-baseline).abs().max());assert error<=1e-6
                report['hook_max_error']=max(report['hook_max_error'],error)
                arrays=dict(captured,hook_error=np.asarray(error),predicted_actions=post(pred).cpu().numpy()[0],video_pts=pts)
                for v,key in enumerate(config.camera_keys):
                    mean=torch.tensor(config.image_mean).view(1,3,1,1);std=torch.tensor(config.image_std).view(1,3,1,1)
                    rgb=(cpu[key]*std+mean)[0].permute(1,2,0).clamp(0,1).numpy()
                    arrays[f'rgb_{v}']=(rgb[32:224]*255).round().astype(np.uint8)
                    if label=='competitive':
                        logits=torch.from_numpy(arrays[f'logits_{v}'])/config.afa.temperature
                        routing=logits.softmax(0).numpy();pooling=routing/(routing.sum(-1,keepdims=True)+1e-8)
                        report['controlled_routing_max_error']=max(report['controlled_routing_max_error'],float(np.abs(routing-arrays[f'routing_{v}']).max()))
                        report['controlled_pooling_max_error']=max(report['controlled_pooling_max_error'],float(np.abs(pooling-arrays[f'attention_{v}'].reshape(16,256)).max()))
                        arrays[f'independent_{v}']=logits.softmax(-1).numpy().reshape(16,16,16)
                np.savez_compressed(out/'arrays'/f"{label}_{r['id']}.npz",**arrays)
                print(label,r['id'],'hook error',error,flush=True)
        for handle in handles:handle.remove()
        for pool,fn in originals:pool._project_cross_attention=fn
        del policy
        torch.cuda.empty_cache()
    assert report['controlled_routing_max_error']<3e-6 and report['controlled_pooling_max_error']<3e-6
    (out/'manifest.json').write_text(json.dumps(report,indent=2))


def render(out):
    from select_clean_full_episodes import montage,overlay,save_pair,with_labels
    from plot_tokenizer_qualitative import COLORS
    from plot_successgrasp_diagnostics import up
    report=json.loads((out/'manifest.json').read_text());records=report['records'];captions=[]
    def winner(rgb,maps):
        w=maps.reshape(16,256).argmax(0).reshape(16,16)
        return np.round(.45*rgb+.55*255*up(COLORS[w],nearest=True)).astype(np.uint8)
    for view,camera in enumerate(['front','wrist']):
        a=[];b=[]
        for r in records:
            with np.load(out/'arrays'/f"competitive_{r['id']}.npz") as z:a.append({k:z[k] for k in z.files})
            with np.load(out/'arrays'/f"trained_no_competitive_{r['id']}.npz") as z:b.append({k:z[k] for k in z.files})
        for x,y in zip(a,b):np.testing.assert_array_equal(x[f'rgb_{view}'],y[f'rgb_{view}'])
        images=[x[f'rgb_{view}'] for x in a];times=[r['frame']/30 for r in records]
        rows=[images,[winner(rgb,x[f'routing_{view}']) for rgb,x in zip(images,a)],
              [winner(rgb,x[f'attention_{view}']) for rgb,x in zip(images,a)],
              [winner(rgb,x[f'independent_{view}']) for rgb,x in zip(images,a)]]
        name=f'{camera}_normalization_winners';image=montage(rows)
        save_pair(image,out/'clean'/name);save_pair(with_labels(image,['RGB','R comp','A comp','A indep'],times),out/'labeled'/name)
        captions.append((name,'Rows: RGB; argmax of competitive query responsibilities R; argmax of spatially normalized competitive pooling A; argmax of independent spatial softmax from the identical QK scores. Same early-BERT checkpoint and inputs. This isolates normalization, not retraining. Assignments use fixed opacity 0.55 in every row. Native 16×16 patch grid, display crop only.'))
        slots=[0,4,8,12]
        for mode,key,other in [('controlled','independent',a),('trained_models','attention',b)]:
            rows=[images]
            for dataset,field in [(a,'attention'),(other,key)]:
                for k in slots:rows.append([overlay(rgb,x[f'{field}_{view}'][k]) for rgb,x in zip(images,dataset)])
            name=f'{camera}_{mode}_slot_maps';image=montage(rows)
            save_pair(image,out/'clean'/name)
            save_pair(with_labels(image,['RGB']+[f'C S{k:02d}' for k in slots]+[f'N S{k:02d}' for k in slots],times),out/'labeled'/name)
            caution=('Same QK scores; changing only normalization. The independent branch is a diagnostic counterfactual, not a separately trained policy.' if mode=='controlled' else 'Real trained checkpoints: single-head early-BERT with competition versus 12-head VLI without competition, both final checkpoints. Architecture/interaction also differ, so this is not an isolated competitive-routing ablation. Maps are geometric pooling distributions; multi-head semantic pooling has additional head-specific weights. Query IDs are not aligned semantic identities across checkpoints.')
            captions.append((name,f'Rows: RGB; competitive maps for fixed queries {slots}; independent maps for the same numeric query indices. All use common log2 attention lift 1×–8× and opacity 0.65. '+caution))
        rows=[images,[winner(rgb,x[f'attention_{view}']) for rgb,x in zip(images,a)],[winner(rgb,x[f'attention_{view}']) for rgb,x in zip(images,b)]]
        name=f'{camera}_trained_models_winners';image=montage(rows);save_pair(image,out/'clean'/name)
        save_pair(with_labels(image,['RGB','A comp','A noComp'],times),out/'labeled'/name)
        captions.append((name,'Rows: RGB; argmax of geometric pooling maps from the competitive single-head early-BERT model; argmax of geometric pooling maps from the noncompetitive 12-head VLI model. Both colored maps are post-hoc winners based on spatially normalized A, not a native R versus A comparison. Models differ in more than routing; query colors do not establish cross-model object correspondence.'))
        for j,(r,x,y) in enumerate(zip(records,a,b)):
            for label,maps in [('competitive',x[f'attention_{view}']),('controlled_independent',x[f'independent_{view}']),('trained_no_competitive',y[f'attention_{view}'])]:
                tiles=[overlay(images[j],maps[k]) for k in range(16)]
                save_pair(montage([tiles[i:i+4] for i in [0,4,8,12]]),out/'all_slots'/f"{camera}_{r['id']}_{label}")
    (out/'CAPTIONS.md').write_text('# Appendix captions\n\n'+'\n\n'.join(f'## {n}\n\n{c}' for n,c in captions)+'\n\nColumns: '+ '; '.join(f"task {r['task_id']}, ep {r['episode']}, frame {r['frame']}" for r in records)+'. All-slot atlases use IDs 0–15 in row-major order; no maps omitted. Query subset 0/4/8/12 chosen a priori, not ranked for favorable appearance.\n')
    gallery=['# Routing: confronto per l’appendice','', '[Metodo e limiti](README.md) · [Didascalie](CAPTIONS.md)','']
    for name,caption in captions:gallery += [f'## {name}','',f'[PDF](clean/{name}.pdf) · [Versione con etichette](labeled/{name}.png)','',f'![{name}](clean/{name}.png)','']
    gallery += ['[Atlanti completi di tutti i 16 slot](all_slots/)']
    (out/'GALLERY.md').write_text('\n'.join(gallery))
    readme='''# Competitive routing: figure per l’appendice

[Galleria](GALLERY.md) · [Didascalie](CAPTIONS.md)

La figura aggiuntiva consigliata è `front_controlled_slot_maps.pdf`, accanto a `front_normalization_winners.pdf`: la prima mostra concentrazione e sovrapposizione di quattro query fisse, la seconda distingue le assegnazioni prima/dopo normalizzazione spaziale. La versione al polso e gli atlanti di tutti i 16 slot sono materiale di supporto.

## Confronto controllato della normalizzazione

Sul medesimo checkpoint early-BERT competitivo si estraggono gli score S query–patch durante il forward reale. Si confrontano R=softmax_query(S/tau), A_comp=R/(sum_patch R+epsilon) e A_ind=softmax_patch(S/tau), a temperatura tau=0.5. La figura dei vincitori usa la stessa opacità fissa in tutte le righe, senza confondere intensità di attention e confidenza. L’argmax di R coincide con quello degli score S; l’argmax di A_comp può cambiare dopo la rinormalizzazione per query.

A_ind è una trasformazione diagnostica degli stessi score: non è un nuovo checkpoint addestrato senza competizione, né una misura di performance della policy con routing disabilitato. Nessuna predizione viene sostituita o eseguita sul robot. I pesi di pooling competitivi e le responsabilità ricostruite sono verificati contro quelli realmente registrati; gli output con e senza registrazione sono confrontati.

## Confronto tra checkpoint realmente addestrati

Si usa anche il checkpoint finale a 160000 micro-step `turbovla_afa_mh12_vli_no_competitive_multitask_full_80k_20261116`, con 12 head e VLI. Quello competitivo finale usa invece single-head ed early-BERT. Le tavole `trained_models` confrontano queste due varianti sulle identiche osservazioni: non isolano causalmente il competitive routing. Per il confronto omologo dei vincitori si usa argmax delle mappe geometriche A in entrambi i modelli. Nella variante multi-head le mappe geometriche sono fuse dalle head e non esauriscono i pesi del pooling semantico. Non si assume corrispondenza semantica fra query di uguale indice nei due modelli.

## Dati e stile

Sei osservazioni, due per task: cubo e peluche dalle demo verificate, peg dai rollout successgrasp già analizzati. I frame provengono dalla precedente selezione temporale e sono identici per ogni confronto; non sono fasi annotate. La natura diversa delle registrazioni è riportata nel manifest. Scala comune 1×–8× rispetto all’uniforme su 256 patch; quattro ID 0/4/8/12 fissati prima del confronto e tutti i 16 disponibili negli atlanti. Nessun k-means, riassegnazione degli ID o segmentazione annotata. Crop del padding solo nel display; immagini clean senza testo, didascalie separate e versioni labeled per revisione. Gli array, gli score e i checkpoint SHA-256 sono nel manifest. Le mappe non provano segmentazione per oggetto o un miglioramento delle prestazioni.
'''
    (out/'README.md').write_text(readme)
    print('Rendered',len(captions),'main pairs and',len(list((out/'all_slots').glob('*.png'))),'all-slot atlases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--extract',action='store_true');a=p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    if a.extract:extract(a.output)
    else:render(a.output)
