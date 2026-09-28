"""FPBench force-error component for an e2IP checkpoint (paired-forces route of FPBench).

Static energy/force predictions for every structure of a dataset (MatPES-PBE or OMat24
rattled-1000), standardized with FPBench's ``build_force_results`` and summarized with the
table builders of ``Force_error/scripts/force_error_metrics.py`` using the thresholds of
``Force_error/analysis/force_error_analysis_matpes_pbe.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

from md22nop.data.graph import collate, from_numpy, make_graph, to_numpy
from md22nop.data.matpes import iter_structures, select
from md22nop.training.train import load_model

FDFT_MIN = 0.01
DF_THRESHOLDS = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5]
ANGLE_THRESHOLDS = [1, 20]
FDFT_SUBSET_THRESHOLDS = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0]
FDFT_THETA_THRESHOLDS = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
DF_LT_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10]
DF_FRAC_THRESH_LARGE = [0.5, 1, 2, 3, 4, 5, 7, 10]
ABS_THRESH = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5]
REL_THRESH = [0.01, 0.05, 0.10, 0.20, 0.3, 0.4, 0.50, 1, 2]
THETA_THRESHOLDS = [1, 5, 10, 20, 30, 60, 90, 120, 178, 180]


def _worker_init():
    torch.set_num_threads(1)


def _g(args):
    i, z, pos, cell, e, f = args
    return i, to_numpy(make_graph(z, pos, cell, r_max=5.0))


def predict_forces(model, arrays, device, max_atoms=6000, workers=8):
    n = len(arrays["n_atoms"])
    preds, energies = [None] * n, np.zeros(n)
    batch, idx, n_at = [], [], 0
    t0 = time.time()

    def flush():
        data = collate(batch)
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
        out = model(data)
        f = out["forces"].detach().double().cpu().numpy()
        e = out["energy"].detach().double().cpu().numpy()
        off = 0
        for j, i in enumerate(idx):
            na = int(arrays["n_atoms"][i])
            preds[i] = f[off:off + na]
            energies[i] = e[j]
            off += na

    # spawn, not fork: the parent has already run torch (model on GPU), and forked children that
    # use torch's thread pool can deadlock (observed on the Vast.ai instance after 60k structures).
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init) as pool:
        for k, (i, g) in enumerate(pool.imap(_g, iter_structures(arrays), chunksize=32)):
            na = int(arrays["n_atoms"][i])
            if batch and n_at + na > max_atoms:
                flush()
                batch, idx, n_at = [], [], 0
            batch.append(from_numpy(g))
            idx.append(i)
            n_at += na
            if k % 20000 == 0:
                print(f"  {k}/{n} structures, {time.time() - t0:.0f}s", flush=True)
        if batch:
            flush()
    return preds, energies


def metrics_tables(force_results, fpbench_dir: Path):
    sys.path.insert(0, str(fpbench_dir / "Force_error" / "scripts"))
    import force_error_metrics as fem

    all_results = {
        m: {"F_dft": d["dft_force_magnitude"], "F_fp": d["fp_force_magnitude"],
            "deltaF": d["force_magnitude_error"], "deltaTheta": d["force_angle_error"],
            "e_vec": d["force_vector_error"]}
        for m, d in force_results.items()
    }
    out = {}
    mae, rmse = fem.build_dF_mae_rmse_fdft_subset(all_results, FDFT_SUBSET_THRESHOLDS)
    out["dF_mae_by_fdft_min"], out["dF_rmse_by_fdft_min"] = mae, rmse
    mae, rmse = fem.build_theta_mae_rmse_fdft_subset(all_results, FDFT_THETA_THRESHOLDS)
    out["theta_mae_by_fdft_min"], out["theta_rmse_by_fdft_min"] = mae, rmse
    mae, rmse = fem.build_dF_mae_rmse_smalldF_subset(all_results, DF_LT_THRESHOLDS, fdft_min=FDFT_MIN)
    out["dF_mae_by_dF_max"], out["dF_rmse_by_dF_max"] = mae, rmse
    out["frac_highly_accurate_dF"] = fem.build_highly_accurate_force_fraction_table(all_results, DF_THRESHOLDS, fdf_min=FDFT_MIN)
    joint = fem.build_joint_dF_theta_accuracy_table(all_results, DF_THRESHOLDS, ANGLE_THRESHOLDS, fdf_min=FDFT_MIN)
    for ac, tbl in joint.items():
        out[f"frac_joint_dF_and_theta_lt_{ac}deg"] = tbl
    out["frac_large_dF"] = fem.build_large_force_error_fraction_table(all_results, DF_FRAC_THRESH_LARGE, fdf_min=FDFT_MIN)
    pa, pb = fem.build_far_from_equilibrium_regime_panels(
        all_results, abs_thresh=ABS_THRESH, rel_thresh=REL_THRESH, threshold=1.0, fdf_min=FDFT_MIN)
    out["fe_panel_A_near_eq_abs"], out["fe_panel_B_fe_rel"] = pa, pb
    out["frac_theta"] = fem.build_angle_accuracy_fraction_table(all_results, THETA_THRESHOLDS, fdf_min=FDFT_MIN)
    return {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="npz with flat arrays (see md22nop.data.matpes)")
    p.add_argument("--name", required=True, help="dataset label, e.g. matpes_pbe or omat24_rattled_1000")
    p.add_argument("--model-name", default="e2IP-NequIP-MatPES10")
    p.add_argument("--exclude-ids", default=None, help="split manifest json; also report held-out-only metrics")
    p.add_argument("--fpbench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-atoms", type=int, default=6000)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.checkpoint, device)
    arrays = dict(np.load(args.dataset))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    preds, energies = predict_forces(model, arrays, device, args.max_atoms, args.workers)
    np.savez_compressed(out / f"{args.name}_predictions.npz", forces=np.concatenate(preds),
                        energies=energies, n_atoms=arrays["n_atoms"], structure_ids=arrays["structure_ids"])

    sys.path.insert(0, str(Path(args.fpbench) / "Force_error" / "scripts"))
    from force_results import build_force_results

    offsets = np.concatenate([[0], np.cumsum(arrays["n_atoms"])])
    dft = [arrays["forces"][offsets[i]:offsets[i + 1]] for i in range(len(preds))]
    sids = [int(s) for s in arrays["structure_ids"]]
    populations = {"all": np.arange(len(preds))}
    if args.exclude_ids:
        m = json.loads(Path(args.exclude_ids).read_text())
        used = set(m["train_structure_ids"]) | set(m["val_structure_ids"])
        populations["held_out"] = np.array([i for i, s in enumerate(sids) if s not in used])

    summary = {"dataset": args.name, "model": args.model_name, "n_structures": len(preds)}
    for pop, idx in populations.items():
        fr = build_force_results([dft[i] for i in idx], {args.model_name: [preds[i] for i in idx]},
                                 structure_ids=[sids[i] for i in idx])
        if pop == "all":
            with open(out / f"{args.name}_force_results_standardized.json", "w") as f:
                json.dump({"schema_version": "1.0", "dataset_name": args.name,
                           "units": {"force": "eV/Å", "angle": "degree"},
                           "models": {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in fr.items()}}, f)
        summary[pop] = {"n_structures": int(len(idx)), "tables": metrics_tables(fr, Path(args.fpbench))}
    (out / f"{args.name}_force_metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    print("FORCE_METRICS")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
