# DCT-QR theory revision summary

## Added scientific ablations

`dct_qr_theory` now exposes six one-factor switches without changing the default full method:

- `no_opponent_term`
- `uniform_step`
- `no_global_coset`
- `no_gain_normalization`
- `no_spatial_icm`
- `no_sync_search`

Use `scripts/run_ablation_study.py --method dct_qr_theory --attack-suite common20`.

## Added curated common-20 attack suite

Exactly 20 attacks, balanced as 5 categories x 4 attacks:

- compression / quantization;
- noise;
- filtering / enhancement;
- geometric / resampling;
- photometric / point processing.

The severe real-world, deformation, structured-loss, compound, stress and extended suites remain separate.

## Added numerical illustrations

`scripts/run_dct_qr_theory_numerics.py` produces theorem diagnostics and the common-20 result table.
Included Lenna illustration:

- PSNR: 53.6035 dB
- clean NC: 1.000000
- common-20 mean NC: 0.975482
- q10 NC: 0.953995
- minimum NC: 0.731832

The strongest single-host ablation effects are:

- no gain normalization: mean NC 0.864249;
- no spatial ICM: mean NC 0.931251;
- no synchronization search: mean NC 0.964952.

These values are numerical illustrations, not multi-host statistical claims.

## Integration fixes

- `dct_qr_theory` is accepted by `run_proposal_benchmark.py`.
- `dct_qr_theory` is accepted by `run_ablation_study.py`.
- `three_method_utils.py` can construct its configuration.
- Added default before/after config files.
- Documentation now reports 9 public proposals rather than 8.
- Added regression tests for the new ablations and exact common-20 grouping.

## Validation

Focused modified-area tests pass. The full repository test collection still stops at the pre-existing `tests/test_pso.py` import of the missing `particle_swarm_maximize`; this revision does not modify that unrelated legacy PSO issue.
