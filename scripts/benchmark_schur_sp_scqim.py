from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'src'))

from qr64_certified import DCT_SCHUR_RESCUE, DirectSchurRescueConfig, embed_proposal, extract_proposal
from qr64_certified.attacks import apply_attack
from qr64_certified.attacks.presets import moderate_attacks
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr

parser=argparse.ArgumentParser()
parser.add_argument('--hosts', nargs='*', default=[])
parser.add_argument('--step', type=float, default=8.5)
parser.add_argument('--output', type=Path, default=ROOT/'results/schur_sp_scqim/reproduced.json')
args=parser.parse_args()
all_hosts=sorted((ROOT/'data/host').glob('*.bmp'))
if args.hosts:
    wanted={x if x.endswith('.bmp') else x+'.bmp' for x in args.hosts}
    all_hosts=[p for p in all_hosts if p.name in wanted]
wm=load_watermark_binary(ROOT/'data/watermark/wm.png',size=64)
cfg=DirectSchurRescueConfig(step=args.step)
rows=[]
for hp in all_hosts:
    t=time.time();host=load_host_rgb(hp)
    watermarked,key,em=embed_proposal(DCT_SCHUR_RESCUE,host,wm,config=cfg,return_metadata=True)
    clean,cm=extract_proposal(watermarked,key,return_metadata=True)
    attacks=[]
    for attack in moderate_attacks():
        rec,meta=extract_proposal(apply_attack(watermarked,attack),key,return_metadata=True)
        attacks.append({'attack':attack.name,'nc':float(nc(wm,rec)),'selected_candidate':meta.get('selected_candidate'),'copy_agreement':meta.get('copy_agreement')})
    vals=[x['nc'] for x in attacks]
    row={'host':hp.name,'psnr':float(psnr(host,watermarked)),'clean_nc':float(nc(wm,clean)),'mean_nc':float(np.mean(vals)),'min_nc':float(np.min(vals)),'attacks':attacks,'seconds':time.time()-t,'embed_metadata':em}
    rows.append(row)
    partial={'method':'dct_schur_rescue','scientific_name':'SP-SCQIM','step':args.step,'host_count':len(rows),'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(partial,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('attacks','embed_metadata')}),flush=True)
summary={'method':'dct_schur_rescue','scientific_name':'SP-SCQIM','step':args.step,'host_count':len(rows),'mean_psnr':float(np.mean([r['psnr'] for r in rows])) if rows else None,'mean_clean_nc':float(np.mean([r['clean_nc'] for r in rows])) if rows else None,'mean_nc':float(np.mean([r['mean_nc'] for r in rows])) if rows else None,'mean_worst_nc':float(np.mean([r['min_nc'] for r in rows])) if rows else None,'global_worst_nc':float(min([r['min_nc'] for r in rows])) if rows else None,'rows':rows}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)
