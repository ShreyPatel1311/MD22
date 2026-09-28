"""FPBench ion-migration NEB component for an e2IP checkpoint.

Executes the code cells of FPBench's own ``Ion_migration_NEB/generation/fp_neb_generation_and_run.ipynb``
with two edits: the configuration cell enables job generation, and the registry cell registers only
this model. The generated per-pathway ``run.py`` scripts (MatCalc RelaxCalc + climbing-image NEBCalc,
unchanged) are run locally in parallel instead of via SLURM, then the notebook's merge and validation
cells write ``ion_migration_neb_fp_results.json``. Leaderboard metrics come from FPBench's
``scripts/export_neb_leaderboard.py``.

The ``dft_static_on_fp_neb`` protocol needs new VASP calculations and is not run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

    jobs = sorted(Path(ns["OUTPUT_BASE"]).glob(f"*/{args.reg_key}/chunk_*/*/run.py"))
    jobs = [j for j in jobs if not (j.parent / "output.json").exists()]
    if args.max_pathways:
        jobs = jobs[: args.max_pathways]
    print(f"running {len(jobs)} job scripts with {args.workers} workers", flush=True)
    env = dict(os.environ, PYTHONPATH=str(Path(args.src).resolve()) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t0 = time.time()

    def run(job):
        with open(job.parent / "stdout.log", "w") as fh:
            rc = subprocess.run([sys.executable, job.name], cwd=job.parent, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        return job, rc

    with ThreadPoolExecutor(args.workers) as ex:
        for k, (job, rc) in enumerate(ex.map(run, jobs)):
            print(f"  [{k + 1}/{len(jobs)}] rc={rc} {job.parent.parent.parent.parent.name}/{job.parent.name} ({time.time() - t0:.0f}s)", flush=True)

    if args.max_pathways:
        return
    for i in sorted(cells):
        if i >= merge_idx:
            exec(compile(cells[i], f"<neb cell {i}>", "exec"), ns)
    merged = Path(ns["MERGED_OUT_DIR"]) / "ion_migration_neb_fp_results.json"
    shutil.copy(merged, out / "ion_migration_neb_fp_results.json")

    rc = subprocess.run([sys.executable, "scripts/export_neb_leaderboard.py", "--reference",
                         "data/ion_migration_neb_reference.json.gz", "--results", str(merged),
                         "--component-dir", ".", "--force"], cwd=work).returncode
    if rc == 0:
        shutil.copy(work / "data" / "ion_migration_neb_leaderboard_summary.json", out / "neb_leaderboard_summary.json")
    print("export rc", rc)


if __name__ == "__main__":
    main()
