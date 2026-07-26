#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Check whether independent SP-SCQIM dominates the previous Schur-gain result.')
parser.add_argument('--candidate',type=Path,default=ROOT/'results/schur_sp_scqim/sp_scqim_13host_step9.json')
parser.add_argument('--reference',type=Path,default=ROOT/'results/redesigned_13host_validation.json')
parser.add_argument('--no-fail',action='store_true',help='Report failed gates without returning a nonzero exit code.')
args=parser.parse_args()
candidate=json.loads(args.candidate.read_text(encoding='utf-8'))
reference_all=json.loads(args.reference.read_text(encoding='utf-8'))
reference=reference_all['dct_schur_rescue']['aggregate']
checks={
 '13 hosts': int(candidate['host_count'])==13,
 'clean NC equals 1': float(candidate['mean_clean_nc'])>=1.0-1e-12,
 'mean PSNR improves': float(candidate['mean_psnr'])>float(reference['psnr']),
 'mean attacked NC improves': float(candidate['mean_nc'])>float(reference['mean_nc']),
 'mean worst NC improves': float(candidate['mean_worst_nc'])>float(reference['mean_worst_nc']),
 'global worst NC improves': float(candidate['global_worst_nc'])>float(reference['global_worst_nc']),
}
print('SP-SCQIM promotion gate')
print(f"candidate: PSNR={candidate['mean_psnr']:.6f}, NC={candidate['mean_nc']:.6f}, worst={candidate['global_worst_nc']:.6f}")
print(f"reference: PSNR={reference['psnr']:.6f}, NC={reference['mean_nc']:.6f}, worst={reference['global_worst_nc']:.6f}")
for name,passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
passed=all(checks.values())
print('PROMOTED' if passed else 'NOT PROMOTED')
if not passed and not args.no_fail:
    raise SystemExit(1)
