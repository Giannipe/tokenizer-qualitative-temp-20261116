#!/usr/bin/env python
"""All-frame/all-slot replay of explicitly pinned verified demonstration pairs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import analyze_successgrasp_diagnostics as extractor
from analyze_successgrasp_fullqk import full_layer_hook
from tokenizer_qualitative import sha256


def run(args):
    original_reader=extractor.read_episodes
    extractor.CompleteRecorder.layer_hook=full_layer_hook
    for study in args.studies:
        source=json.loads((study/'source.json').read_text())
        indices=source['episode_indices']
        def selected_reader(root):
            selected=[ep for ep in original_reader(root) if ep['episode_index'] in indices]
            assert [ep['episode_index'] for ep in selected]==indices
            assert all(ep['tasks']==[source['instruction']] for ep in selected)
            assert sum(ep['length'] for ep in selected)==source['expected_frames']
            return selected
        extractor.read_episodes=selected_reader
        kind=json.loads((args.checkpoint/'config.json').read_text())['type']
        root=study/kind;root.mkdir(exist_ok=True)
        provenance=dict(schema='verified_demo_full_diagnostics_v1',wrapper_sha256=sha256(Path(__file__)),
            base_source_sha256=sha256(Path(extractor.__file__)),fullqk_source_sha256=sha256(Path(__file__).with_name('analyze_successgrasp_fullqk.py')),
            full_matrix_qk_all_dino_layers=True,selected_episode_indices=indices,source=source)
        file=root/'verified_extractor_provenance.json'
        if file.exists():assert json.loads(file.read_text())==provenance
        file.write_text(json.dumps(provenance,indent=2))
        print('Starting',source['task_slug'],kind,indices,'frames',source['expected_frames'],flush=True)
        extractor.run(SimpleNamespace(study=study,checkpoint=args.checkpoint,device=args.device,threads=args.threads,limit_frames=args.limit_frames))
        if args.limit_frames:break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies',type=Path,nargs='+',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--limit-frames',type=int)
    run(parser.parse_args())
