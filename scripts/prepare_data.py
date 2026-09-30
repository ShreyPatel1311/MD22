"""Download and convert the datasets used for training and FPBench evaluation.

    python scripts/prepare_data.py matpes --out data/raw
        MatPES-PBE v2025.1 -> matpes_pbe_full.npz, 10% random subset (seed 0) split 95/5 into
        matpes_pbe_10pct_train.npz / matpes_pbe_10pct_val.npz, and matpes_pbe_10pct_split.json.
    python scripts/prepare_data.py omat24 --out data/raw
        OMat24 rattled-1000 validation split (117,004 structures) -> omat24_rattled_1000.npz.
        Rows are read with ase_db_backends' LMDBDatabase in select() order, as FPBench's chunker does.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np

from md22nop.data.matpes import (
    MATPES_PBE_URL,
    convert_to_arrays,
    download,
    select,
    subset_split,
    write_split_manifest,
)

OMAT24_RATTLED_1000_VAL_URL = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz"
)


def prepare_matpes(out: Path, fraction: float, seed: int):
    raw = download(MATPES_PBE_URL, out / "MatPES-PBE-2025.1.json.gz")
    full_path = out / "matpes_pbe_full.npz"
    arrays = dict(np.load(full_path)) if full_path.exists() else convert_to_arrays(raw, full_path)
    n = len(arrays["n_atoms"])
    train_idx, val_idx = subset_split(n, fraction=fraction, seed=seed)
    np.savez(out / "matpes_pbe_10pct_train.npz", **select(arrays, train_idx))
    np.savez(out / "matpes_pbe_10pct_val.npz", **select(arrays, val_idx))
    write_split_manifest(out / "matpes_pbe_10pct_split.json", train_idx, val_idx, n, seed, fraction)
    print(f"MatPES-PBE: {n} structures, {int(arrays['n_atoms'].sum())} atoms; "
          f"train={len(train_idx)} val={len(val_idx)}")


def prepare_omat24(out: Path, fraction: float = 1.0, seed: int = 0):
    from ase_db_backends.aselmdb import LMDBDatabase

    tgz = download(OMAT24_RATTLED_1000_VAL_URL, out / "rattled-1000.tar.gz")
    extract = out / "omat24_extracted"
    if not extract.exists():
        with tarfile.open(tgz) as tar:
            tar.extractall(extract)
    dbs = sorted(extract.rglob("*.aselmdb"))
    assert len(dbs) == 1, f"expected one .aselmdb, found {dbs}"
    db = LMDBDatabase(str(dbs[0]))
    numbers, positions, forces, cells, energies, n_atoms = [], [], [], [], [], []
    for row in db.select():
        atoms = row.toatoms()
        numbers.append(atoms.numbers.astype(np.int16))
        positions.append(atoms.positions)
        forces.append(np.asarray(row.forces))
        cells.append(atoms.cell.array)
        energies.append(float(row.energy))
        n_atoms.append(len(atoms))
    arrays = dict(numbers=np.concatenate(numbers), positions=np.concatenate(positions),
                  forces=np.concatenate(forces), cells=np.stack(cells), energies=np.asarray(energies),
                  n_atoms=np.asarray(n_atoms, dtype=np.int64),
                  structure_ids=np.arange(len(n_atoms), dtype=np.int64),
                  matpes_ids=np.asarray([str(i) for i in range(len(n_atoms))]))
    np.savez(out / "omat24_rattled_1000.npz", **arrays)
    print(f"OMat24 rattled-1000 val: {len(n_atoms)} structures")
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(n_atoms), int(round(fraction * len(n_atoms))), replace=False))
        np.savez(out / f"omat24_rattled_1000_{int(round(fraction * 100))}pct.npz", **select(arrays, idx))
        print(f"OMat24 rattled-1000 subset: {len(idx)} structures (fraction {fraction}, seed {seed})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["matpes", "omat24"])
    p.add_argument("--out", default="data/raw")
    p.add_argument("--fraction", type=float, default=0.10, help="subset fraction (MatPES training subset / OMat24 evaluation subset)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "matpes":
        prepare_matpes(out, args.fraction, args.seed)
    else:
        prepare_omat24(out, args.fraction, args.seed)


if __name__ == "__main__":
    main()
