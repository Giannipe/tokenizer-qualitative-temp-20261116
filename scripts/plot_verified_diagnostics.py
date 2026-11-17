#!/usr/bin/env python
"""The complete peg diagnostic layout, applied to pinned verified demo pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import plot_successgrasp_diagnostics as renderer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    original_documents=renderer.documents
    original_browser=renderer.browser_assets
    for study in args.studies:
        source=json.loads((study/'source.json').read_text())
        slug=source['task_slug'];label=source['title'];total=source['expected_frames']
        out=args.output_root/f'{slug}_verified'
        out.mkdir(exist_ok=True,parents=True)
        cache=Path('/tmp')/f'{study.name}_summary.npz'
        marker=cache.with_suffix('.source.json')
        if cache.exists():assert marker.exists() and json.loads(marker.read_text())==source
        marker.write_text(json.dumps(source,indent=2))

        def browser_assets(destination,data,models,episodes):
            original_browser(destination,data,models,episodes)
            path=destination/'viewer/index.html'
            text=path.read_text().replace('Successgrasp',f'{label} · demo verificate').replace('successgrasp_',f'{slug}_verified_')
            text=text.replace('Due episodi · due camere', 'Due dimostrazioni verificate · due camere')
            path.write_text(text)

        def documents(current_study,destination,catalog,selections,report,episodes):
            original_documents(current_study,destination,catalog,selections,report,episodes)
            gallery=destination/'GALLERY.md'
            gallery.write_text(gallery.read_text().replace('# Success-grasp: diagnostica completa',f'# {label}: diagnostica completa delle demo verificate'))
            path=destination/'README.md';old=path.read_text()
            common=old[old.index('## Tensori estratti su ogni frame'):]
            start=common.index('Il calcolo CLS ottimizzato')
            end=common.index('## Significato delle figure')
            common=common[:start]+('In entrambi i replay le diagnostiche di tutti i layer DINO usano la matrice QK completa, con la stessa forma del riferimento eager. Il forward produttivo SDPA resta invariato. Gli hash dell’estrattore e del riuso della pipeline sono conservati in `provenance/`.\n\n')+common[end:]
            common=common.replace('tutti i 706 frame',f'tutti i {total} frame')
            counts='; '.join(f"episodio {eid}: {n} frame" for eid,n in episodes)
            header=f'''# {label}: diagnostica integrale delle demo verificate

[Galleria completa](GALLERY.md) · [Didascalie](CAPTIONS.md) · [Selezione](selection.json) · [Verifiche](validation.json)

## Dati e copertura

Si usano soltanto le due dimostrazioni verificate selezionate per questo task: **{counts}**. I rollout falliti sono esclusi da questa raccolta.

Dataset: [{source['repo_id']}](https://huggingface.co/datasets/{source['repo_id']}), revisione `{source['revision']}`. Istruzione registrata: **{source['instruction']}**. Sono gli stessi due episodi usati nelle precedenti raccolte, qui elaborati interamente, a 30 Hz, con camera frontale e al polso.

**Tutti i {total} frame sono elaborati da ciascun checkpoint**, per {2*total} predizioni di chunk e {4*total} viste complessive fra due policy e due camere. La procedura usa lo stesso tokenizer early-BERT finale AFA e lo stesso TurboVLA VLI fine-tuned dell'analisi peg. Entrambi i checkpoint sono a 160000 micro-step; percorsi e SHA-256 sono nei manifest in `provenance/`.

Queste sono **dimostrazioni registrate del dataset di training**, con comportamento verificato, e non nuovi rollout riusciti delle policy. Il confronto rappresenta un replay offline delle osservazioni, degli stati e dell'istruzione originali. Non si esegue training e nessuna azione viene inviata al robot.

Inferenza in `eval()` e `inference_mode()`, FP32, batch uno, senza AMP/TF32. Per ogni frame si verifica che attivare la registrazione diagnostica non modifichi le predizioni; si controllano anche identità di RGB/stati/comandi fra i due replay, normalizzazione delle distribuzioni, geometria, decomposizione dei token e replay del transformer.

'''
            extra='''\n## Coerenza con peg, cubo e peluche\n\nLe raccolte usano la stessa disposizione dei pannelli, gli stessi colori degli ID degli slot, le stesse scale per attention/routing e gli stessi checkpoint. Le PCA sono adattate separatamente per raccolta e camera: i loro colori non sono categorie semantiche e non vanno confrontati direttamente fra task. Tutti i 16 slot restano presenti, anche se diffusi, inattivi o ridondanti.\n\nLa proiezione dei token dopo il transformer usa il routing originario come ancoraggio grafico dell'ID; dopo il mixing non descrive tutto il campo ricettivo del token.\n\n## Ripetere l'analisi\n\nUsare `scripts/extract_verified_diagnostics.py` per il replay dei soli indici specificati in `source.json`, `scripts/plot_verified_diagnostics.py` per figure e viewer e `scripts/verify_verified_diagnostics.py` per i controlli completi. L'estrazione richiede l'ambiente LeRobot con i pacchetti locali delle due policy e gli encoder disponibili. Il rendering legge gli archivi salvati senza ulteriore inferenza.\n\n'''
            path.write_text(header+common+extra)
            selection=destination/'selection.json';doc=json.loads(selection.read_text())
            doc.update(condition='verified_demo',task_slug=slug,dataset_episode_indices=source['episode_indices'],
                renderer_entrypoint_sha256=digest(Path(__file__)),shared_renderer_sha256=digest(Path(renderer.__file__)))
            selection.write_text(json.dumps(doc,indent=2))
            report_path=destination/'validation.json';report=json.loads(report_path.read_text())
            report.update(condition='verified_demo',task_slug=slug,episode_indices=source['episode_indices'])
            report_path.write_text(json.dumps(report,indent=2))
            provenance=destination/'provenance'
            for kind in ['turbovla_afa','turbovla']:
                shutil.copy2(current_study/kind/'verified_extractor_provenance.json',provenance/f'{kind}_verified_extractor_provenance.json')

        renderer.browser_assets=browser_assets
        renderer.documents=documents
        previous_argv=sys.argv
        try:
            sys.argv=['plot_verified_diagnostics.py',str(study),'--output',str(out)]
            renderer.main()
        finally:
            sys.argv=previous_argv


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',type=Path,nargs='+',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    run(p.parse_args())
