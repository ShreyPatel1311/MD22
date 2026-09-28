"""FPBench phase-stability & elemental-ordering component for an e2IP checkpoint.

Runs FPBench's own ``get_energy`` routine (copied verbatim from
``Phase_stability_ordering/generation/convexhull_ordering_run_generator.ipynb``: fixed-cell ASE FIRE,
fmax = 0.01 eV/A, max 10,000 steps, plus a static evaluation on the DFT-relaxed structure) for the
597 unique hull candidates and 6,100 ordering candidates, fans shared endpoints out exactly as the
generator's merge step does, and scores the result together with the seven published FPs using
``build_combined_hull_table`` / ``build_combined_ordering_table`` / ``build_rmsd_table``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

ENERGY_FUNC = '''
def get_energy(atoms, static=False, fmax=0.01, max_steps=10000):
    atoms = atoms.copy()
    atoms.calc = calc
    if static:
        return {
            "status":      "success",
            "mode":        "static",
            "energy_total": float(atoms.get_potential_energy()),
            "forces":      atoms.get_forces().tolist(),
        }
    from ase.optimize import FIRE
    optimizer = FIRE(atoms, logfile=None)
    converged = optimizer.run(fmax=fmax, steps=max_steps)
    n_steps = optimizer.nsteps
    return {
        "status":            "success" if converged else "non_converged",
        "mode":              "relaxation",
        "energy_total":      float(atoms.get_potential_energy()),
        "forces":            atoms.get_forces().tolist(),
        "relaxed_structure": adaptor.get_structure(atoms).as_dict(),
        "converged":         bool(converged),
        "n_steps":           int(n_steps),
        "fmax_requested":    fmax,
        "max_steps":         max_steps,
        "fixed_cell":        True,
    }
'''
FMAX, MAX_STEPS = 0.01, 10000
_NS = {}


def _init_worker(checkpoint):
    from pymatgen.io.ase import AseAtomsAdaptor
    from md22nop.calculator import E2IPCalculator

    _NS.update({"calc": E2IPCalculator(checkpoint), "adaptor": AseAtomsAdaptor()})
    exec(ENERGY_FUNC, _NS)


def _run(item):
    from pymatgen.core import Structure

    sid, data = item
    adaptor, get_energy = _NS["adaptor"], _NS["get_energy"]
    entry = {k: v for k, v in data.items() if k not in ("initial_structure", "relaxed_structure")}
    try:
        initial_atoms = adaptor.get_atoms(Structure.from_dict(data["initial_structure"]))
        entry["relax"] = get_energy(initial_atoms, static=False, fmax=FMAX, max_steps=MAX_STEPS)
    except Exception as e:
        entry["relax"] = {"status": "failed", "error": str(e)}
    try:
        final_atoms = adaptor.get_atoms(Structure.from_dict(data["relaxed_structure"]))
        entry["static"] = get_energy(final_atoms, static=True)
    except Exception as e:
        entry["static"] = {"status": "failed", "error": str(e)}
    return sid, entry


def build_entries(reference_data):
    """Same candidate population and sid keys as the generator's Sections 5-6."""
    hull, entries, endpoint_seen = reference_data["hull"], {}, set()
    for system in sorted(hull):
        for cid, rec in hull[system].items():
            if rec["role"] == "interior":
                entries[f"{system}||{cid}"] = {
                    "role": "interior", "system": system, "candidate_id": cid,
                    "phase_id": rec["phase_id"], "composition": rec["composition"],
                    "initial_structure": rec["initial_structure"], "relaxed_structure": rec["relaxed_structure"]}
            elif cid not in endpoint_seen:
                endpoint_seen.add(cid)
                entries[f"__endpoint__||{cid}"] = {
                    "role": "endpoint", "candidate_id": cid, "phase_id": rec["phase_id"],
                    "composition": rec["composition"], "endpoint_side": rec["endpoint_side"],
                    "initial_structure": rec["initial_structure"], "relaxed_structure": rec["relaxed_structure"]}
    assert len(entries) == 561 + 36
    for group_key, group in reference_data["ordering"].items():
        for name, rec in group["orderings"].items():
            entries[f"ORD::{group_key}||{name}"] = {
                "role": "ordering", "group_key": group_key, "ordered_name": name, "system": group["system"],
                "phase_id": group["phase_id"], "composition": group["composition"],
                "initial_structure": rec["initial_structure"], "relaxed_structure": rec["relaxed_structure"]}
    assert len(entries) == 597 + 6100
    return entries


def build_fragment(reference_data, results):
    """Generator Section 8 fan-out: one endpoint result is routed to every system it borders."""
    hull_frag = {"relax": defaultdict(dict), "static": defaultdict(dict)}
    for system, candidates in reference_data["hull"].items():
        for cid, rec in candidates.items():
            sid = f"{system}||{cid}" if rec["role"] == "interior" else f"__endpoint__||{cid}"
            for mode in ("relax", "static"):
                hull_frag[mode][system][cid] = dict(results[sid][mode])
    ordering_frag = {"relax": defaultdict(dict), "static": defaultdict(dict)}
    for sid, entry in results.items():
        if sid.startswith("ORD::"):
            group_key, name = sid[5:].split("||", 1)
            for mode in ("relax", "static"):
                ordering_frag[mode][group_key][name] = dict(entry[mode])
    to_plain = lambda d: {m: dict(v) for m, v in d.items()}
    return {"hull": to_plain(hull_frag), "ordering": to_plain(ordering_frag)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fpbench", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-key", default="e2ip_nequip_matpes10")
    p.add_argument("--model-name", default="e2IP-NequIP-MatPES10")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    comp = Path(args.fpbench) / "Phase_stability_ordering"
    sys.path.insert(0, str(comp / "scripts"))
    import convexhull_analysis_utils as cau

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference_data = cau.load_standardized_reference(comp / "data" / "phase_stability_ordering_reference.json.gz")["reference_data"]
    entries = build_entries(reference_data)

    done_path = out / "phase_results.jsonl"
    results = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            sid, entry = json.loads(line)
            results[sid] = entry
    todo = [(sid, e) for sid, e in entries.items() if sid not in results]
    # Longest-first ordering keeps workers busy at the end of the run.
    todo.sort(key=lambda it: -len(it[1]["initial_structure"]["sites"]))
    print(f"{len(results)} done, {len(todo)} to run", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.checkpoint,)) as pool, \
            open(done_path, "a") as fh:
        for k, (sid, entry) in enumerate(pool.imap_unordered(_run, todo)):
            results[sid] = entry
            fh.write(json.dumps([sid, entry]) + "\n")
            fh.flush()
            if k % 200 == 0:
                print(f"  {k + 1}/{len(todo)} ({time.time() - t0:.0f}s) {sid[:60]} relax={entry['relax']['status']}", flush=True)

    fragment = build_fragment(reference_data, results)
    with open(out / "phase_stability_ordering_fragment.json", "w") as f:
        json.dump({"model_key": args.model_key, "protocol": {"relax": {"optimizer": "ASE FIRE", "fmax": FMAX,
                   "max_steps": MAX_STEPS, "fixed_cell": True}}, **fragment}, f)

    published = cau.load_standardized_results(comp / "data" / "phase_stability_ordering_results_standardized.json.gz")
    fp_results = dict(published["models"])
    fp_results[args.model_key] = fragment
    bench = cau.build_phase_stability_ordering_results(reference_data=reference_data, fp_results=fp_results)
    fps = [fp for fp in cau.PHASE_STABILITY_ORDERING_MODEL_ORDER if fp in fp_results] + [args.model_key]
    names = dict(cau.PHASE_STABILITY_ORDERING_MODEL_NAMES)
    names[args.model_key] = args.model_name

    hull_table, hull_rows = cau.build_combined_hull_table(bench["dft_hull"], bench["fp_hull"], fps, names)
    ordering_table, ordering_summaries = cau.build_combined_ordering_table(bench["dft_ordering"], bench["fp_ordering"], fps, names)
    rmsd_table, _, rmsd_summaries = cau.build_rmsd_table(bench["dft_hull"], bench["fp_hull"], fps, names)
    for name, tbl in (("hull", hull_table), ("ordering", ordering_table), ("rmsd", rmsd_table)):
        tbl.to_csv(out / f"phase_{name}_table.csv")
    counts = {
        "relax_status": dict(__import__("collections").Counter(e["relax"]["status"] for e in results.values())),
        "static_status": dict(__import__("collections").Counter(e["static"]["status"] for e in results.values())),
        "ground_state_relax": {k: v for k, v in hull_rows["relax"][args.model_key]["_gs"].items()},
        "rmsd_summary": rmsd_summaries[args.model_key],
    }
    (out / "phase_summary.json").write_text(json.dumps(counts, indent=1, default=str))
    print(hull_table.to_string(), ordering_table.to_string(), rmsd_table.to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
