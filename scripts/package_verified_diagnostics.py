#!/usr/bin/env python
"""Package validated verified-demo figures and one offline viewer per episode."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


def run(args):
    scripts=Path(__file__).parent
    names=['extract_verified_diagnostics.py','plot_verified_diagnostics.py','verify_verified_diagnostics.py',
        'verify_verified_viewer.py','package_verified_diagnostics.py','analyze_successgrasp_diagnostics.py',
        'analyze_successgrasp_fullqk.py','analyze_full_episodes.py','tokenizer_qualitative.py',
        'plot_successgrasp_diagnostics.py','plot_tokenizer_qualitative.py','select_clean_full_episodes.py','successgrasp_viewer.html']
    for name in names:
        shutil.copy2(scripts/name,args.output_root/'scripts'/name)
    sha={name:hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in names}
    for study in args.studies:
        source=json.loads((study/'source.json').read_text())
        out=args.output_root/f"{source['task_slug']}_verified"
        report=json.loads((out/'validation.json').read_text())
        assert report['raw_archives_verified']==source['expected_frames']*2
        assert report['viewer_embedded_frames_verified']==source['expected_frames']
        viewer_report=json.loads((out/'viewer_validation.json').read_text())
        assert viewer_report['all_seven_modes_work'] and not viewer_report['page_errors']
        manifest=json.loads((out/'provenance/turbovla_afa_manifest.json').read_text())
        episodes=[(ep['episode_index'],ep['length']) for ep in manifest['episodes']]
        assert [eid for eid,_ in episodes]==source['episode_indices']
        (out/'provenance/scripts_sha256.json').write_text(json.dumps(sha,indent=2))
        validation=f'''\n## Esito delle verifiche\n\nVerificati tutti i **{report['raw_archives_verified']} archivi frame/policy**. Predizioni con registrazione diagnostica attiva/disattiva: errore massimo **{report['hook_error']}** su tutti i frame. Decomposizione token geometria+semantica: errore massimo **{report['detailed_errors']['fused_sum']}**. Gli altri errori numerici, gli indici completi e gli hash per frame sono in [validation.json](validation.json) e [archive_index.csv](archive_index.csv). Il viewer è collaudato su file locale, incluse tutte le sette modalità, camera al polso, query 15, ultimo frame ed esportazione PNG: [viewer_validation.json](viewer_validation.json).\n\n## Viewer per episodio\n\n'''
        links=[]
        for eid,n in episodes:
            links.append(f'[Episodio {eid}: tutti i {n} frame](viewer_episode_{eid:03d}.zip)')
        validation+=' · '.join(links)+'\n\nEstrai ciascuno ZIP in una cartella separata e apri `viewer/index.html`. Non serve avviare un server. La cartella `viewer/` nella repository comprende entrambi gli episodi insieme.\n'
        readme=out/'README.md';text=readme.read_text()
        if '## Esito delle verifiche' in text:text=text.split('## Esito delle verifiche')[0]
        readme.write_text(text+validation)
        gallery=out/'GALLERY.md';text=gallery.read_text()
        if '## Scarica il viewer' in text:text=text.split('## Scarica il viewer')[0]
        gallery.write_text(text+'\n\n## Scarica il viewer\n\n'+' · '.join(links)+'\n\nEstrai lo ZIP e apri `viewer/index.html`.\n')
        html=(out/'viewer/index.html').read_text()
        for eid,n in episodes:
            path=out/f'viewer_episode_{eid:03d}.zip'
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                episode_html,count=re.subn(r'const episodes=\{[^}]+\}',lambda m:'const episodes='+json.dumps({eid:n}),html,count=1)
                assert count==1
                z.writestr('viewer/index.html',episode_html)
                for file in sorted((out/'viewer/data'/f'ep{eid:03d}').glob('*.js')):z.write(file,file.relative_to(out))
                for file in ['CAPTIONS.md','selection.json','validation.json','viewer_validation.json']:z.write(out/file,file)
                url=f"https://github.com/Giannipe/tokenizer-qualitative-temp-20261116/blob/main/{out.name}"
                z.writestr('README.md',f"# {source['title']} · demo verificata {eid}\n\nApri `viewer/index.html`: include tutti i {n} frame, entrambe le camere e tutti i 16 slot. Nessun server o rete richiesti.\n\n[Metodo]({url}/README.md) · [Figure PNG/PDF]({url}/GALLERY.md)\n")
                z.writestr('GALLERY.md',f'[Galleria PNG/PDF su GitHub]({url}/GALLERY.md)\n')
            assert path.stat().st_size<100*1024**2
            print('Packaged',out.name,eid,n,path.stat().st_size,flush=True)
        broken=[]
        for name in ['README.md','GALLERY.md']:
            for link in re.findall(r'\]\(([^)]+)\)',(out/name).read_text()):
                if not re.match(r'https?://|#',link) and not (out/link).exists():broken.append((name,link))
        assert not broken,broken
        records=json.loads((out/'figure_manifest.json').read_text())
        assert len(records)==report['figure_pairs']
        for record in records:
            assert (out/record['png']).exists() and (out/record['pdf']).exists()
        print('Collection ready',out.name,'figure pairs',len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',type=Path,nargs='+',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    run(p.parse_args())
