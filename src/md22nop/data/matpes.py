"""MatPES-PBE (v2025.1) download, streaming conversion to flat arrays, and the 10% subset split.

Source: https://s3.us-east-1.amazonaws.com/materialsproject-contribs/MatPES_2025_1/MatPES-PBE-2025.1.json.gz
(MatPES repository, github.com/materialyzeai/matpes). ``structure_id`` is the entry's index in the
raw list, which is the identifier FPBench's force chunker assigns (``structure_id == original_index``).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Dict

import ijson
import numpy as np

MATPES_PBE_URL = (
    "https://s3.us-east-1.amazonaws.com/materialsproject-contribs/MatPES_2025_1/MatPES-PBE-2025.1.json.gz"
)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def _items(path: Path):
    with gzip.open(path, "rb") as f:
        first = f.read(1).lstrip()
    prefix = "item" if first == b"[" else "data.item"
    with gzip.open(path, "rb") as f:
        yield from ijson.items(f, prefix, use_float=True)


def _structure_arrays(sd: Dict):
    lattice = np.asarray(sd["lattice"]["matrix"], dtype=np.float64)
    numbers, positions = [], []
    from ase.data import atomic_numbers

    for site in sd["sites"]:
        species = site["species"]
        if len(species) != 1:
            raise ValueError("disordered site")
        numbers.append(atomic_numbers[species[0]["element"]])
        positions.append(site["xyz"])
    return np.asarray(numbers, dtype=np.int16), np.asarray(positions, dtype=np.float64), lattice


def convert_to_arrays(json_gz: Path, out_npz: Path) -> Dict[str, np.ndarray]:
    """Streams the raw MatPES json.gz into flat arrays (one row per atom / per structure)."""
    numbers, positions, forces, cells, energies, n_atoms, ids = [], [], [], [], [], [], []
    for idx, entry in enumerate(_items(json_gz)):
        z, pos, cell = _structure_arrays(entry["structure"])
        f = np.asarray(entry["forces"], dtype=np.float64)
        assert f.shape == pos.shape, f"entry {idx}: forces/positions mismatch"
        numbers.append(z)
        positions.append(pos)
        forces.append(f)
        cells.append(cell)
        energies.append(float(entry["energy"]))
        n_atoms.append(len(z))
        ids.append(str(entry.get("matpes_id", idx)))
    arrays = {
        "numbers": np.concatenate(numbers),
        "positions": np.concatenate(positions),
        "forces": np.concatenate(forces),
        "cells": np.stack(cells),
        "energies": np.asarray(energies),
        "n_atoms": np.asarray(n_atoms, dtype=np.int64),
        "structure_ids": np.arange(len(n_atoms), dtype=np.int64),
        "matpes_ids": np.asarray(ids),
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrays)
    return arrays


def subset_split(n_structures: int, fraction: float = 0.10, val_fraction: float = 0.05, seed: int = 0):
    """Random ``fraction`` of all structures, split into train/val. Returns sorted index arrays."""
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(n_structures)[: int(round(fraction * n_structures))]
    n_val = int(round(val_fraction * len(chosen)))
    return np.sort(chosen[n_val:]), np.sort(chosen[:n_val])


def select(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    offsets = np.concatenate([[0], np.cumsum(arrays["n_atoms"])])
    atom_idx = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in idx])
    out = {k: arrays[k][idx] for k in ("cells", "energies", "n_atoms", "structure_ids", "matpes_ids")}
    out.update({k: arrays[k][atom_idx] for k in ("numbers", "positions", "forces")})
    return out


def iter_structures(arrays: Dict[str, np.ndarray]):
    offsets = np.concatenate([[0], np.cumsum(arrays["n_atoms"])])
    for i in range(len(arrays["n_atoms"])):
        a, b = offsets[i], offsets[i + 1]
        yield i, arrays["numbers"][a:b], arrays["positions"][a:b], arrays["cells"][i], arrays["energies"][i], arrays["forces"][a:b]


def write_split_manifest(path: Path, train_idx, val_idx, n_total: int, seed: int, fraction: float):
    path.write_text(json.dumps({
        "source_url": MATPES_PBE_URL,
        "n_structures_total": int(n_total),
        "fraction": fraction,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_structure_ids": [int(i) for i in train_idx],
        "val_structure_ids": [int(i) for i in val_idx],
    }))
