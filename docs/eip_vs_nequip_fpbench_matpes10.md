# eIP-NequIP vs. plain NequIP on FPBench (MatPES-PBE 10% subset)

This report compares two models on the four FPBench components:

- **NequIP-MatPES10**: a plain NequIP interatomic potential.
- **eIP-NequIP-MatPES10**: the same NequIP backbone with the evidential head of eIP added.

The two models share data, backbone, optimizer, schedule and evaluation. Only the uncertainty head and the force loss differ. Each model was trained once, with one seed.

Code: branch `claude/gallant-volta-9wfu97` of this repository.

- Plain NequIP run configuration: commit `a9b7856`.
- eIP-NequIP run configuration: commit `4ab4a46`.

## 1. Sources

| Item | Source |
|---|---|
| eIP method | Xu et al., *Evidential Deep Learning for Interatomic Potentials*, [arXiv:2407.13994](https://arxiv.org/abs/2407.13994); Nature Communications, [doi:10.1038/s41467-025-67663-y](https://www.nature.com/articles/s41467-025-67663-y) |
| eIP reference code | [github.com/xuhan323/eIP](https://github.com/xuhan323/eIP), commit `04190fb52b1ee67207ada6e0562ed8e469c8a6b1` (`PaiNN.py`: `update_u`; `run.py`: `quant_evi_loss`) |
| Benchmark | FPBench, [github.com/mogroupumd/FPBench](https://github.com/mogroupumd/FPBench), commit `b8475daeab17b76404b0cde1cbe53ca1a60a92e5` |
| Training data | MatPES-PBE v2025.1 (`MatPES-PBE-2025.1.json.gz`, URL in `src/md22nop/data/matpes.py`) |
| OMat24 evaluation data | OMat24 `val/rattled-1000` (URL in `scripts/prepare_data.py`) |
| Backbone library | `nequip==0.19.1` |
| Loss weighting convention | `mace-torch` 0.3.16 defaults (`mace/tools/arg_parser.py`): `--energy_weight` 1.0, `--stage_two_energy_weight` 1000.0, `--forces_weight` 100.0 |

## 2. Training methodology

### 2.1 Data
Data preparation is done by `scripts/prepare_data.py matpes` and `src/md22nop/data/matpes.py`.

- **Full dataset:** MatPES-PBE v2025.1 has 434,712 structures and 3,881,535 atoms.
- **Subset:** a random 10% subset with seed 0, split 95/5 into:
  - **training:** 41,297 structures;
  - **validation / held-out:** 2,174 structures. These were never used for gradient updates.
- **Graphs:** neighbour lists use a 5.0 Å cutoff with periodic boundaries (`src/md22nop/data/graph.py`).
- **Training-set statistics:**
  - per-element energy shifts from a least-squares fit of total energies to element counts;
  - per-atom energy scale equal to the force RMS, 1.850 eV/Å;
  - average number of neighbours: 31.89.

### 2.2 Backbone (identical for both models)
The backbone mirrors the `FullNequIPGNNModel` of nequip 0.19.1 with the "M" preset (`src/md22nop/models/e2ip.py`, `_nequip_energy_model`):

- 4 interaction layers, l_max = 2, parity off;
- features [128, 64, 32] for l = 0, 1, 2;
- type embedding of size 32;
- radial MLP with 1 layer of width 128;
- 8 Bessel functions with a polynomial cutoff (p = 6).

The per-atom energy readout is followed by the fixed per-type scale and shift. Forces are F = −∂E/∂x, and stress comes from the strain derivative.

### 2.3 Models

| | NequIP-MatPES10 | eIP-NequIP-MatPES10 |
|---|---|---|
| Parameters | 3,189,824 | 3,194,179 |
| Head | none | eIP `update_u`: Linear(64→64) → ShiftedSoftplus → Linear(64→3), applied to each Cartesian component (x, y, z) of the l = 1 features of the second-to-last interaction layer |
| Outputs per atom | E, F | E, F, and (ν, α, β) for each of x, y, z: ν = softplus + 1e-5, α = softplus + 1 + 1e-5, β = softplus |
| Force prediction | F = −∇E | γ = F = −∇E |

Notes on the eIP port (`src/md22nop/models/eip.py`):
- **Feature layer:** the published eIP code reads the vector features of PaiNN's last layer. NequIP's last layer is scalar-only, so the head reads the `64x1o` features of the second-to-last layer.
- **Component ordering:** in e3nn 0.6.0 the `1o` components are ordered (x, y, z). The test `test_eip_vector_features_rotate_and_loss_backward` checks that the extracted features rotate as R·v.
- **Equivariance:** the head applies a nonlinearity to each component separately, as the reference code does. The predicted (ν, α, β) are therefore not rotation-equivariant. The forces are unaffected.

### 2.4 Loss functions
Both models use the same energy term:

L_E = λ_E · mean over structures of ((E_pred − E_DFT) / N_atoms)²

- λ_E = 1.
- λ_E rises to 1000 at 75% of the 5 h schedule (225 min). Training stopped at 120 min, so the 1000 weight was never reached.

The force terms differ:

| Model | Force term | Weight |
|---|---|---|
| NequIP | mean over atoms and components of (F_pred − F_DFT)² | 100 (mace-torch default `--forces_weight`) |
| eIP-NequIP | Σ over x, y, z of the mean over atoms of `quant_evi_loss` (quantile q = 0.6, regularizer coefficient 0.1), ported unchanged from the reference `run.py`, including its clamps | 1 |

Deviations from the eIP reference, applied so that only the head and force loss differ between the two models:
- **Energy loss:** the reference code uses an L1 total-energy loss and weights `10000 · L_force + 0.1 · L_energy`. The per-atom MSE energy term and weighting above were used instead.
- **Regularizer:** the reference code's regularizer is ρ_q(y − γ)·(2ν + α + 1/β)·(y − γ). This has one more (y − γ) factor than Eq. (9) of the paper. The code was followed.
- **Clamps:** the reference code clamps the per-component NLL to ≥ 1e-6 and the regularizer to ≥ 1e-6. Components at these bounds contribute no gradient.

### 2.5 Optimization (identical for both models)
Implemented in `src/md22nop/training/train.py`:

- **Batching:** batches are packed to at most 1,200 atoms and reshuffled every epoch.
- **Optimizer:** AdamW with peak learning rate 1e-3 and weight decay 1e-3. Gradient norms are clipped at 10, and a step is skipped if the loss or gradient norm is non-finite.
- **Learning rate:**
  - linear warm-up over the first 2% of a 5 h wall-time budget;
  - then cosine decay to 1% of the peak;
  - `--hours 5 --stop-after-min 120`, so training stopped after 120 min of that 5 h schedule.
- **EMA:** exponential moving average of the weights with decay 0.999. Validation and all evaluations use the EMA weights.
- **Precision:** float32, with TF32 disabled.
- **Validation and checkpoints:**
  - The model is scored on the 2,174 held-out structures every 30 min.
  - The checkpoint saved at the 120-min stop (`final.pt`) is used for all FPBench evaluations.

### 2.6 Compute

| | NequIP-MatPES10 | eIP-NequIP-MatPES10 |
|---|---|---|
| Hardware | Vast.ai, 1 × RTX 4090 | Vast.ai, 1 × RTX 4090 |
| Container image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | same |
| Training (UTC, 2026-09-29) | 16:36:03–18:36:14 | 20:47:26–22:47:46 |
| Epochs / steps at stop | 121 / 37,059 | 115 / 35,119 |
| Phase stability evaluation | 43 min | 46 min |
| NEB evaluation (15 pathways) | 8 min | 6 min |

## 3. Evaluation protocol
One checkpoint per model is evaluated on all four components with no fine-tuning. The subsets are identical for both models. The pipeline is in `scripts/vast_pipeline.sh`.

| Component | Data | Protocol |
|---|---|---|
| Force error, MatPES held-out | 2,174-structure held-out split (§2.1) | Static predictions, standardized with FPBench `build_force_results`. Tables come from `Force_error/scripts/force_error_metrics.py`. Only atoms with \|F_DFT\| > 0.01 eV/Å are counted (`scripts/eval_force.py`). |
| Force error, OMat24 | Random 10% (seed 0) of OMat24 rattled-1000 validation: 11,700 of 117,004 structures | Same as above |
| Phase stability & ordering | Full FPBench reference set: 6,697 relaxations | Fixed-cell ASE FIRE, fmax 0.01 eV/Å, ≤ 10,000 steps. Static evaluation on the DFT-relaxed structures. FPBench `convexhull_analysis_utils` tables (`scripts/eval_phase.py`). |
| Ion-migration NEB | Random 10% (seed 0) of the 154 active pathways: the same 15 pathways for both models | FPBench generator notebook, unchanged: MatCalc `RelaxCalc` endpoints (fmax 0.002 eV/Å, ≤ 500 steps), then climbing-image NEB with 5 intermediate images (fmax 0.05 eV/Å, ≤ 1,000 steps). Scored with FPBench `export_neb_leaderboard.build_leaderboard` on the 15-pathway subset (`scripts/eval_neb.py`). |

Definitions used in the force tables (from the docstrings in FPBench `Force_error/scripts/force_error_metrics.py`):
- **FE atoms** (far from equilibrium): atoms with \|F_DFT\| > 1 eV/Å.
- **Excluding large-error atoms:** atoms with Δ\|F\| < 1 eV/Å.
- **Joint accuracy:** the percentage of atoms with Δ\|F\| < 0.01 eV/Å and Δθ below 1° or below 20°.

## 4. Results

### 4.1 Validation during training (2,174 held-out MatPES structures, EMA weights)

| Minutes | NequIP energy MAE (meV/atom) | NequIP force MAE (meV/Å) | eIP-NequIP energy MAE (meV/atom) | eIP-NequIP force MAE (meV/Å) |
|---|---|---|---|---|
| 30 | 543.1 | 278.9 | 220.9 | 272.4 |
| 60 | 410.3 | 232.9 | 132.3 | 220.5 |
| 90 | 318.3 | 217.2 | 108.0 | 205.9 |
| 120 (final) | 264.4 | 211.2 | 97.4 | 202.2 |

### 4.2 Force error, MatPES-PBE held-out (2,174 structures)
Units: Δ\|F\| in eV/Å, Δθ in degrees, fractions in % of atoms.

| Metric | NequIP | eIP-NequIP |
|---|---|---|
| Δ\|F\| MAE / RMSE | 0.306 / 2.307 | 0.290 / 2.059 |
| Fraction Δ\|F\| > 1 eV/Å | 4.05 | 3.49 |
| FE atoms: Δ\|F\| MAE / RMSE | 0.608 / 4.473 | 0.586 / 3.991 |
| FE atoms: Δθ MAE / RMSE | 18.4 / 26.4 | 17.5 / 24.8 |
| Fraction Δ\|F\| < 0.01 eV/Å | 4.06 | 4.19 |
| Joint accuracy (Δθ < 1° / < 20°) | 0.68 / 1.98 | 0.45 / 2.20 |
| Δθ MAE / RMSE | 40.9 / 61.2 | 39.7 / 60.4 |
| Excluding large-error atoms: Δ\|F\| MAE / RMSE | 0.216 / 0.296 | 0.208 / 0.289 |

### 4.3 Force error, OMat24 rattled-1000 (10%, 11,700 structures)

| Metric | NequIP | eIP-NequIP |
|---|---|---|
| Δ\|F\| MAE / RMSE | 1.056 / 4.761 | 1.008 / 5.036 |
| Fraction Δ\|F\| > 1 eV/Å | 23.81 | 21.35 |
| FE atoms: Δ\|F\| MAE / RMSE | 1.264 / 5.318 | 1.208 / 5.625 |
| FE atoms: Δθ MAE / RMSE | 12.0 / 18.8 | 10.4 / 15.6 |
| Fraction Δ\|F\| < 0.01 eV/Å | 1.56 | 1.83 |
| Joint accuracy (Δθ < 1° / < 20°) | 0.03 / 1.03 | 0.02 / 1.32 |
| Δθ MAE / RMSE | 17.2 / 28.0 | 14.8 / 23.9 |
| Excluding large-error atoms: Δ\|F\| MAE / RMSE | 0.355 / 0.443 | 0.322 / 0.411 |

### 4.4 Phase stability and ordering (full set)
All 6,697 relaxations and all 6,697 static calculations completed successfully for both models.

**Hull stability.** Columns: energy error (meV/atom) / ground-state agreement % / within-phase hull-minimum agreement % / hull-minimum agreement %.

| Mode | NequIP | eIP-NequIP |
|---|---|---|
| Full FP relaxation | 152 / 45.3 / 33.9 / 13.6 | 162 / 43.5 / 35.5 / 13.6 |
| Static on DFT-relaxed structures | 151 / 43.5 / 41.3 / 27.3 | 158 / 45.3 / 41.3 / 13.6 |

Ground-state agreement after full relaxation corresponds to 73/161 compositions for NequIP and 70/161 for eIP-NequIP.

**Ordering.** Columns: energy error (meV/atom) / Top-1 % / Recall@3 % / Recall@10 % / Spearman ρ / ranking errors % / mean/max ΔE_DFT (meV/atom).

| Mode | NequIP | eIP-NequIP |
|---|---|---|
| Full FP relaxation | 162 / 5.6 / 16.6 / 52.6 / 0.07 / 47.8 / 14/42 | 127 / 5.6 / 17.6 / 53.3 / 0.07 / 47.3 / 14/42 |
| Static on DFT-relaxed structures | 156 / 15.4 / 25.8 / 59.0 / 0.23 / 41.8 / 11/36 | 130 / 15.1 / 29.1 / 57.7 / 0.20 / 42.3 / 11/32 |

**Relaxed-structure RMSD vs. DFT**

| Metric | NequIP | eIP-NequIP |
|---|---|---|
| Map success (%) | 87.8 | 88.3 |
| Mean / max RMSD (Å) | 0.49 / 1.87 | 0.39 / 1.73 |
| RMSD < 0.05 / 0.10 / 0.20 Å (%) | 5.9 / 12.4 / 26.0 | 8.2 / 20.5 / 39.8 |

### 4.5 Ion-migration NEB (15-pathway subset, seed 0)

| Metric | NequIP | eIP-NequIP |
|---|---|---|
| Non-converged pathways | 0 / 15 | 0 / 15 |
| Barrier MAE / RMSE, full FP-NEB (eV) | 0.181 / 0.213 | 0.199 / 0.238 |
| Barrier MAE / RMSE, static on DFT images (eV) | 0.179 / 0.230 | 0.179 / 0.238 |
| Endpoint ΔE MAE / RMSE (eV) | 0.073 / 0.115 | 0.064 / 0.096 |
| Endpoint ranking agreement (%) | 66.7 | 73.3 |
| Energy-profile shape agreement (%) | 73.3 | 86.7 |
| Endpoint RMSD mean / max (Å) | 0.173 / 0.327 | 0.152 / 0.426 |
| Endpoint RMSD < 0.05 / 0.10 / 0.20 Å (%) | 0 / 23.3 / 53.3 | 0 / 40.0 / 76.7 |

### 4.6 Column count
The table below counts, for each component, the columns in §4.2–4.5 where one model has the better value. The phase count excludes the ΔE_DFT column.

| Component | eIP-NequIP better | NequIP better | Tied |
|---|---|---|---|
| MatPES held-out forces (14 columns) | 13 | 1 | 0 |
| OMat24 10% forces (14) | 11 | 3 | 0 |
| Phase stability & ordering (26) | 14 | 8 | 4 |
| NEB, 15 pathways (14) | 7 | 5 | 2 |

Columns where NequIP has the better value:
- **Forces:** joint accuracy at 1° (MatPES and OMat24); OMat24 Δ\|F\| RMSE; OMat24 FE-atom Δ\|F\| RMSE.
- **Phase:** hull energy error (both modes); full-relaxation ground-state agreement; static hull-minimum agreement; static Top-1, Recall@10, Spearman ρ and ranking-error rate.
- **NEB:** barrier MAE/RMSE (full) and barrier MAE/RMSE (static; the MAE is 0.1792 vs. 0.1794 eV); endpoint RMSD maximum.

## 5. Limitations
- **Seeds:** one training run per model. No repeat runs were made, so the size of run-to-run variation is not known.
- **Training scope:** 10% of MatPES-PBE and 120 min of training, stopped before the end of the learning-rate schedule and before the stage-two energy weight.
- **Evaluation subsets:** NEB uses 15 of 154 pathways and OMat24 uses 10% of rattled-1000. These results are not comparable with published FPBench leaderboard rows computed on the full sets.
- **Loss choices:** the eIP energy loss and loss weighting differ from the eIP reference code (§2.4). eIP's head reads a different layer than in the original PaiNN implementation (§2.3).
- **Uncertainty not assessed:** the eIP uncertainty outputs (ν, α, β) were not evaluated in this report.
- **Train/test overlap:** whether structures in the phase-stability or NEB reference sets overlap with the MatPES training split was not checked.
- **Checkpoints:** both trained checkpoints were deleted together with their Vast.ai instances. Only the metrics above were kept.
