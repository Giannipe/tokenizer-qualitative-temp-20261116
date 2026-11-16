import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess,re

parser=argparse.ArgumentParser(description='Verify full episode diagnostics and clean figure exports')
parser.add_argument('study',type=Path)
parser.add_argument('--figures',type=Path,required=True)
parser.add_argument('--arrays-only',action='store_true')
args=parser.parse_args()
study=args.study
out=args.figures
checks={}
manifest={kind:json.loads((study/kind/'manifest.json').read_text()) for kind in ('turbovla_afa','turbovla')}
pre=[]
for kind in manifest:
    pre.append(json.loads((Path(manifest[kind]['checkpoint'])/'policy_preprocessor.json').read_text()))
assert pre[0]==pre[1], 'Saved preprocessing pipelines differ'
checks['identical_saved_preprocessing']=True
max_routing=0.;max_coords=0.;totals={k:0 for k in manifest}
y,x=np.meshgrid(np.linspace(-1,1,16),np.linspace(-1,1,16),indexing='ij')
grid=np.stack([x.ravel(),y.ravel()],axis=-1)
for kind in manifest:
    files=sorted((study/kind).glob('*.npz'))
    assert len(files)==12
    for file in files:
        with np.load(file) as data:
            n=len(data['frame_index']);totals[kind]+=n
            np.testing.assert_array_equal(data['frame_index'],np.arange(n))
            assert data['predicted_actions'].shape==(n,12,6)
            for key in data.files:
                value=data[key]
                if value.dtype.kind in 'fc':assert np.isfinite(value).all(),(file,key)
            if kind=='turbovla_afa':
                assert data['rgb'].shape==(n,2,192,256,3)
                max_routing=max(max_routing,float(np.abs(data['routing'].sum(2)-1).max()))
                reconstruction=data['attention'].reshape(n,2,16,256).astype(float)@grid
                max_coords=max(max_coords,float(np.abs(reconstruction-data['keypoints']).max()))
    coverage=json.loads((study/kind/'coverage.json').read_text())
    assert totals[kind]==coverage['expected_frames']==coverage['completed_frames']==4117
    assert coverage['hook_checks']==36 and coverage['hook_max_error']==0
assert max_routing<2e-6 and max_coords<3e-6,(max_routing,max_coords)
checks.update(episodes_per_policy=12,complete_frames_per_policy=totals,routing_sum_max_error=max_routing,keypoint_reconstruction_max_error=max_coords,all_arrays_finite=True,hook_checks_per_policy=36,hook_prediction_max_difference=0.0)
print(json.dumps(checks,indent=2),flush=True)
if args.arrays_only:sys.exit(0)
selection=json.loads((out/'selection.json').read_text())
assert len(selection['selection'])==24
assert selection['total_frames_per_policy']==4117
for entry in selection['selection']:
    frames=entry['frames'];slots=entry['slots']
    assert len(frames)==8 and len(set(frames))==8 and frames==sorted(frames)
    assert len(slots)==len(set(slots))==4
    with np.load(out/'selected_arrays'/f"{entry['id']}.npz") as data:
        np.testing.assert_array_equal(data['frame_index'],frames)
        for kind,rows in [('tokens',6),('policies',5)]:
            file=out/'clean'/f"{entry['id']}_{kind}.png"
            with Image.open(file) as image:
                assert image.size==(2069,192*rows+3*(rows-1))
                assert all(abs(dpi-300)<.1 for dpi in image.info['dpi'])
                pixels=np.array(image)
                for c in range(8):
                    np.testing.assert_array_equal(pixels[:192,c*259:c*259+256],data['rgb'][c,entry['view']])
                for c in range(7):assert np.all(pixels[:,c*259+256:c*259+259]==255)
            info=subprocess.check_output(['pdfinfo',str(file.with_suffix('.pdf'))],text=True)
            assert re.search(r'Pages:\s+1\b',info)
            width=float(re.search(r'Page size:\s+([0-9.]+)',info).group(1))
            assert abs(width-2069/300*72)<.01
for folder in ('clean','minimal_labels'):
    pngs=list((out/folder).glob('*.png'));pdfs=list((out/folder).glob('*.pdf'))
    assert len(pngs)==len(pdfs)==48
    for path in pngs:
        with Image.open(path) as image:image.verify()
    for path in pdfs:
        info=subprocess.check_output(['pdfinfo',str(path)],text=True)
        assert re.search(r'Pages:\s+1\b',info)
with (out/'all_frame_slot_metrics.csv').open() as stream:
    count=sum(1 for _ in csv.DictReader(stream))
assert count==4117*2*16
checks.update(clean_png_pdf_pairs=48,minimal_labels_png_pdf_pairs=48,selected_camera_sequences=24,frames_per_strip=8,metrics_rows=count,crop_is_display_only=True,figure_top_row_matches_selected_rgb=True,figure_gutters_white=True,figure_outer_margins_px=0)
(out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Figure checks completed:',json.dumps(checks,indent=2))
