"""FPBench ion-migration NEB component for an e2IP checkpoint.

Executes the code cells of FPBench's own ``Ion_migration_NEB/generation/fp_neb_generation_and_run.ipynb``
with two edits: the configuration cell enables job generation, and the registry cell registers only
this model. The generated per-pathway ``run.py`` scripts (MatCalc RelaxCalc + climbing-image NEBCalc,
unchanged) are run locally in parallel instead of via SLURM, then the notebook's merge and validation
cells write ``ion_migration_neb_fp_results.json``. Leaderboard metrics come from FPBench's
``scripts/export_neb_leaderboard.py``.

The ``dft_static_on_fp_neb`` protocol needs new VASP calculations and is not run.

With ``--fraction`` < 1 only a seeded random subset of the 154 pathways is run. The leaderboard
metrics are then computed by the same ``export_neb_leaderboard.build_leaderboard`` on reference and
results files restricted to that subset; only its fixed 154-pathway population check and fixed
model-name table are bypassed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REGISTRY_CELL = '''
POTENTIAL_REGISTRY = {
    "__REG_KEY__": {
        "output_key":    "__OUTPUT_KEY__",
        "display_name":  "__DISPLAY_NAME__",
        "site_pkgs":     "__SITE_PKGS__",
        "venv_activate": "__VENV__",
        "model_path":    "__MODEL_PATH__",
        "import_lines":  "from md22nop.calculator import E2IPCalculator",
        "calc_lines":    'calc = E2IPCalculator(MODEL_PATH)',
    },
}
ACTIVE_FP_KEYS = ["__REG_KEY__"]
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fpbench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--src", required=True, help="path to this repo's src/ directory")
    p.add_argument("--reg-key", default="e2ip_nequip_matpes10")
    p.add_argument("--output-key", default="e2IP_NequIP_MatPES10")
    p.add_argument("--display-name", default="e2IP-NequIP-MatPES10")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-pathways", type=int, default=None, help="smoke-test: run only the first N jobs")
    p.add_argument("--fraction", type=float, default=1.0, help="random fraction of the 154 pathways to run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out).resolve()
    work = out / "neb_component"
    if not work.exists():
        shutil.copytree(Path(args.fpbench) / "Ion_migration_NEB", work)
    gen_dir = work / "generation"
    nb = json.loads((gen_dir / "fp_neb_generation_and_run.ipynb").read_text())
    cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}

    registry = (REGISTRY_CELL.replace("__REG_KEY__", args.reg_key).replace("__OUTPUT_KEY__", args.output_key)
                .replace("__DISPLAY_NAME__", args.display_name).replace("__SITE_PKGS__", str(Path(args.src).resolve()))
                .replace("__VENV__", sys.prefix).replace("__MODEL_PATH__", str(Path(args.checkpoint).resolve())))
    config = cells[4].replace("GENERATE_JOBS           = False", "GENERATE_JOBS           = True")
    assert "GENERATE_JOBS           = True" in config
    registry_idx = next(i for i, s in cells.items() if s.lstrip().startswith("POTENTIAL_REGISTRY = {"))
    merge_idx = next(i for i, s in cells.items() if "merged_out_path = MERGED_OUT_DIR" in s)
    submit_md = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown" and "## 11. Submission Scripts" in "".join(c["source"]))
    submit_idx = min(i for i in cells if i > submit_md)

    os.chdir(gen_dir)
    ns = {"__name__": "__fpbench_neb__"}
    for i in sorted(cells):
        if i >= merge_idx or i == submit_idx:
            continue
        src = config if i == 4 else registry if i == registry_idx else cells[i]
        exec(compile(src, f"<neb cell {i}>", "exec"), ns)

    all_keys = sorted(ns["reference_data"]["common_pathway_keys"])
    if args.fraction < 1.0:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        subset = sorted(all_keys[i] for i in rng.choice(len(all_keys), int(round(args.fraction * len(all_keys))), replace=False))
    else:
        subset = all_keys
    (out / "neb_pathway_subset.json").write_text(json.dumps({"fraction": args.fraction, "seed": args.seed, "pathways": subset}))
    print(f"pathways: {len(subset)} of {len(all_keys)}", flush=True)
    safe_subset = {k.replace("|", "_p") for k in subset}
    jobs = sorted(Path(ns["OUTPUT_BASE"]).glob(f"*/{args.reg_key}/chunk_*/*/run.py"))
    jobs = [j for j in jobs if j.parent.name in safe_subset]
    jobs = [j for j in jobs if not (j.parent / "output.json").exists()]
    if args.max_pathways:
        jobs = jobs[: args.max_pathways]
    print(f"running {len(jobs)} job scripts with {args.workers} workers", flush=True)
    env = dict(os.environ, PYTHONPATH=str(Path(args.src).resolve()) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    # One CPU thread per job: with the library defaults, `workers` parallel jobs each start a
    # thread per core, and the oversubscribed CPU made each force call take seconds (GPU idle).
    env.update({v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    t0 = time.time()

    def run(job):
        with open(job.parent / "stdout.log", "w") as fh:
            rc = subprocess.run([sys.executable, job.name], cwd=job.parent, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        return job, rc

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(run, job) for job in jobs]
        for k, fut in enumerate(as_completed(futures)):
            job, rc = fut.result()
            print(f"  [{k + 1}/{len(jobs)}] rc={rc} {job.parent.parent.parent.parent.name}/{job.parent.name} ({time.time() - t0:.0f}s)", flush=True)

    if args.max_pathways:
        return
    for i in sorted(cells):
        if i >= merge_idx:
            exec(compile(cells[i], f"<neb cell {i}>", "exec"), ns)
    merged = Path(ns["MERGED_OUT_DIR"]) / "ion_migration_neb_fp_results.json"
    shutil.copy(merged, out / "ion_migration_neb_fp_results.json")

    subset_leaderboard(work, merged, subset, len(all_keys), args, out)


def subset_leaderboard(work: Path, merged: Path, subset, n_all: int, args, out: Path):
    """FPBench's build_leaderboard on reference/results restricted to ``subset`` pathways."""
    import gzip

    ref = json.load(gzip.open(work / "data" / "ion_migration_neb_reference.json.gz", "rt"))
    res = json.loads(merged.read_text())
    keep = set(subset)
    ref["pathways"] = {k: v for k, v in ref["pathways"].items() if k in keep}
    ref["common_pathway_keys"] = [k for k in ref["common_pathway_keys"] if k in keep]
    for model in res["models"].values():
        for proto in model.values():
            if isinstance(proto, dict):
                for field in ("pathways", "unsuccessful_pathways"):
                    if field in proto:
                        proto[field] = {k: v for k, v in proto[field].items() if k in keep}
    ref_path, res_path = out / "neb_reference_subset.json", out / "neb_results_subset.json"
    ref_path.write_text(json.dumps(ref))
    res_path.write_text(json.dumps(res))

    sys.path.insert(0, str(work / "scripts"))
    import export_neb_leaderboard as exp
    import neb_analysis as na
    from neb_plots import area_between_curves, simplify_class

    exp.EXPECTED_ACTIVE_PATHWAYS = len(subset)
    if len(subset) != n_all:
        exp.validate_active_population = lambda reference_data: None
    exp.FP_DISPLAY_NAMES[args.output_key] = args.display_name
    leaderboard = exp.build_leaderboard(str(ref_path), str(res_path), na, area_between_curves, simplify_class,
                                        fp_keys=[args.output_key])
    leaderboard["pathway_subset"] = {"fraction": args.fraction, "seed": args.seed, "n": len(subset)}
    (out / "neb_leaderboard_summary.json").write_text(json.dumps(leaderboard, indent=1))
    print("NEB_LEADERBOARD")
    print(json.dumps(leaderboard["models"], indent=1))
    return leaderboard


if __name__ == "__main__":
    main()
