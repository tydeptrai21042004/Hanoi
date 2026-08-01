#!/usr/bin/env python3
"""Validate the active QR-conditioned pairwise coset DCT-QIM proposal."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qr64_certified.attacks import apply_attack
from qr64_certified.attacks.presets import moderate_attacks
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import nc, psnr
from qr64_certified.proposals.config import QR64Config
from qr64_certified.proposals.method import embed, extract

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--host-limit',type=int,default=0)
    parser.add_argument('--output',type=Path,default=ROOT/'results/dct_qr_pairwise_coset_13host.json')
    args=parser.parse_args(); os.environ.setdefault('JILP_NUM_THREADS','1')
    cfg=QR64Config.from_mapping(json.loads((ROOT/'configs/dct_qr_after_abc.json').read_text()))
    if not cfg.coset_optimization_enabled or cfg.coset_group_size != 2:
        raise RuntimeError('Active DCT-QR config is not the pairwise coset proposal.')
    hosts=sorted((ROOT/'data/host').glob('*.bmp'),key=lambda p:p.name.lower())
    if args.host_limit: hosts=hosts[:args.host_limit]
    wm=load_watermark_binary(ROOT/'data/watermark/wm.png',size=64); attacks=moderate_attacks(); rows=[]
    for path in hosts:
        host=load_host_rgb(path); marked,key=embed(host,wm,config=cfg); clean,meta=extract(marked,key,return_metadata=True)
        av={a.name:float(nc(wm,extract(apply_attack(marked,a),key))) for a in attacks}; vals=np.asarray(list(av.values()))
        row={'host':path.name,'psnr':float(psnr(host,marked)),'clean_nc':float(nc(wm,clean)),'mean_nc':float(vals.mean()),'q10_nc':float(np.quantile(vals,.1)),'min_nc':float(vals.min()),'attacks':av,'coset_group_size':int(meta['coset_group_size']),'coset_group_count':int(meta['coset_group_count'])}
        rows.append(row); print(f"{path.name}: PSNR={row['psnr']:.6f}, clean={row['clean_nc']:.6f}, meanNC={row['mean_nc']:.6f}, minNC={row['min_nc']:.6f}",flush=True)
    report={'method':'DCT-QR Pairwise Coset-Optimized Gain-Normalized QIM','configuration':cfg.to_dict(),'protocol':{'hosts':[p.name for p in hosts],'watermark':'data/watermark/wm.png','watermark_size':[64,64],'attacks':[a.name for a in attacks]},'hosts':rows,'aggregate':{'host_count':len(rows),'attack_count':len(attacks),'mean_psnr':float(np.mean([r['psnr'] for r in rows])),'clean_nc':float(np.mean([r['clean_nc'] for r in rows])),'mean_attacked_nc':float(np.mean([r['mean_nc'] for r in rows])),'mean_q10_nc':float(np.mean([r['q10_nc'] for r in rows])),'mean_worst_nc':float(np.mean([r['min_nc'] for r in rows])),'global_worst_nc':float(np.min([r['min_nc'] for r in rows]))}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report['aggregate'],indent=2))
if __name__=='__main__': main()
