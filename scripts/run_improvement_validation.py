from __future__ import annotations
import csv, json, os, time
from pathlib import Path
import numpy as np

from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import psnr, nc
from qr64_certified.attacks import apply_attack, AttackConfig
from qr64_certified.attacks.presets import moderate_attacks
from qr64_certified.config import QR64Config
from qr64_certified.method import embed as embed_qr, extract as extract_qr
from qr64_certified.direct_schur_rescue import (
    DirectSchurRescueConfig,
    embed as embed_schur,
    extract as extract_schur,
    extract_components,
)
from qr64_certified.cd_detqr import (
    CDDetQRConfig,
    embed_cd_detqr,
    extract_cd_detqr,
)
from improvement_experiments import (
    embed_dct_qr_adaptive,
    extract_dct_qr_adaptive,
    embed_sparse_joint_schur,
    extract_sparse_joint_schur,
    extract_spatial_affine_sync,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'improvement_validation'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('JILP_NUM_THREADS', '1')

wm = load_watermark_binary(ROOT / 'data/watermark/wm.png', size=64)
all_host_paths = sorted((ROOT / 'data/host').glob('*.bmp'), key=lambda p:p.name.lower())
start = int(os.environ.get('HOST_START', '0')); end = int(os.environ.get('HOST_END', str(len(all_host_paths))))
host_paths = all_host_paths[start:end]
attacks = moderate_attacks()
geometry_attacks = [
    AttackConfig('rotation_2deg', 'rotation', {'degrees':2.0}),
    AttackConfig('shear_0p08', 'shear', {'shear_x':0.08}),
]

def eval_attacks(out, key, extractor, attacks_list):
    vals=[]; per={}
    for a in attacks_list:
        rec = extractor(apply_attack(out,a), key)
        v=float(nc(wm,rec)); vals.append(v); per[a.name]=v
    return vals,per

rows=[]
qr_cfg=QR64Config.from_mapping(json.loads((ROOT/'configs/dct_qr_after_pso.json').read_text()))
schur_cfg=DirectSchurRescueConfig.from_mapping(json.loads((ROOT/'configs/dct_schur_rescue_after_pso.json').read_text()))
sp_raw=json.loads((ROOT/'configs/spatial_cd_detqr_after_pso.json').read_text())
sp_before_cfg=CDDetQRConfig(**sp_raw)
sp_after_raw=dict(sp_raw); sp_after_raw['pilot_count']=71
sp_after_cfg=CDDetQRConfig(**sp_after_raw)

for hp in host_paths:
    host=load_host_rgb(hp)
    print('HOST',hp.name,flush=True)

    # DCT-QR before
    out,key=embed_qr(host,wm,config=qr_cfg)
    vals,per=eval_attacks(out,key,extract_qr,attacks)
    rows.append({'host':hp.name,'method':'DCT-QR','stage':'before','psnr':psnr(host,out),'clean_nc':nc(wm,extract_qr(out,key)),'mean_nc':np.mean(vals),'min_nc':np.min(vals),**{f'nc_{k}':v for k,v in per.items()}})
    # DCT-QR adaptive after: weakest 20%=20, middle 40%=15, strongest 40%=12
    out,key=embed_dct_qr_adaptive(host,wm,config=qr_cfg,step_levels=(20.0,15.0,12.0),fractions=(0.20,0.60))
    vals,per=eval_attacks(out,key,extract_dct_qr_adaptive,attacks)
    rows.append({'host':hp.name,'method':'DCT-QR','stage':'after_adaptive','psnr':psnr(host,out),'clean_nc':nc(wm,extract_dct_qr_adaptive(out,key)),'mean_nc':np.mean(vals),'min_nc':np.min(vals),**{f'nc_{k}':v for k,v in per.items()}})

    # DCT-Schur before
    out,key=embed_schur(host,wm,config=schur_cfg)
    vals,per=eval_attacks(out,key,extract_schur,attacks)
    comp=extract_components(out,key)
    schur_clean=(np.asarray(comp['schur_map'])>0).astype(np.uint8)*255
    rows.append({'host':hp.name,'method':'DCT-Schur Rescue','stage':'before','psnr':psnr(host,out),'clean_nc':nc(wm,extract_schur(out,key)),'mean_nc':np.mean(vals),'min_nc':np.min(vals),'secondary_clean_nc':nc(wm,schur_clean),**{f'nc_{k}':v for k,v in per.items()}})
    # Sparse joint Schur candidate
    out,key,meta=embed_sparse_joint_schur(host,wm,config=schur_cfg,rescue_count=128,joint_iters=20,schur_step=0.25,selection_mode='primary_confidence')
    vals,per=eval_attacks(out,key,extract_sparse_joint_schur,attacks)
    rows.append({'host':hp.name,'method':'DCT-Schur Rescue','stage':'after_sparse_joint','psnr':psnr(host,out),'clean_nc':nc(wm,extract_sparse_joint_schur(out,key)),'mean_nc':np.mean(vals),'min_nc':np.min(vals),'secondary_selected_clean_accuracy':meta['history'][-1]['selected_accuracy_after_primary_closure'],'joint_iters_used':len(meta['history']),**{f'nc_{k}':v for k,v in per.items()}})

    # Spatial before
    out,key=embed_cd_detqr(host,wm,config=sp_before_cfg)
    vals,per=eval_attacks(out,key,extract_cd_detqr,attacks)
    gvals,gper=eval_attacks(out,key,extract_cd_detqr,geometry_attacks)
    rows.append({'host':hp.name,'method':'Spatial CD-DetQR','stage':'before','psnr':psnr(host,out),'clean_nc':nc(wm,extract_cd_detqr(out,key)),'mean_nc':np.mean(vals),'min_nc':np.min(vals),'geometry_mean_nc':np.mean(gvals),**{f'nc_{k}':v for k,v in per.items()},**{f'nc_{k}':v for k,v in gper.items()}})
    # Spatial after: 71 pilots + affine pilot search. Restrict test grid to the declared +/-2 deg and +/-0.08 shear candidates.
    out,key=embed_cd_detqr(host,wm,config=sp_after_cfg)
    def spatial_after(im,k):
        return extract_spatial_affine_sync(im,k,angle_grid=(-2.0,0.0,2.0),shear_grid=(-0.08,0.0,0.08))
    vals,per=eval_attacks(out,key,extract_cd_detqr,attacks)
    gvals,gper=eval_attacks(out,key,spatial_after,geometry_attacks)
    rows.append({'host':hp.name,'method':'Spatial CD-DetQR','stage':'after_affine_sync','psnr':psnr(host,out),'clean_nc':nc(wm,spatial_after(out,key)),'mean_nc':np.mean(vals),'min_nc':np.min(vals),'geometry_mean_nc':np.mean(gvals),**{f'nc_{k}':v for k,v in per.items()},**{f'nc_{k}':v for k,v in gper.items()}})

fieldnames=sorted({k for r in rows for k in r})
with (OUT/f'per_host_{start}_{end}.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=fieldnames);wr.writeheader();wr.writerows(rows)

print(f'WROTE {len(rows)} rows for hosts {start}:{end}')
