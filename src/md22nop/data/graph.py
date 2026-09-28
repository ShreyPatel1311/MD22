"""ASE Atoms / raw arrays -> NequIP ``AtomicDataDict`` graphs with neighbor lists."""

from __future__ import annotations

from typing import List, Optional, Sequence

import ase
import ase.data
import numpy as np
import torch
from nequip.data import AtomicDataDict, compute_neighborlist_

# Element symbols H (Z=1) .. Pu (Z=94) as NequIP type names.
TYPE_NAMES: List[str] = list(ase.data.chemical_symbols[1:95])
_Z_TO_TYPE = torch.full((119,), -1, dtype=torch.long)
_Z_TO_TYPE[1:95] = torch.arange(94)


def make_graph(
    numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool] = (True, True, True),
    r_max: float = 5.0,
    energy: Optional[float] = None,
    forces: Optional[np.ndarray] = None,
    dtype: torch.dtype = torch.float32,
) -> AtomicDataDict.Type:
    z = torch.as_tensor(np.asarray(numbers), dtype=torch.long)
    types = _Z_TO_TYPE[z]
    if (types < 0).any():
        raise ValueError(f"unsupported atomic numbers: {sorted(set(z[types < 0].tolist()))}")
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.as_tensor(np.asarray(positions), dtype=dtype),
        AtomicDataDict.CELL_KEY: torch.as_tensor(np.asarray(cell), dtype=dtype).view(1, 3, 3),
        AtomicDataDict.PBC_KEY: torch.as_tensor(np.asarray(pbc), dtype=torch.bool).view(1, 3),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: z,
        AtomicDataDict.ATOM_TYPE_KEY: types,
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([len(z)], dtype=torch.long),
    }
    if energy is not None:
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = torch.tensor([[float(energy)]], dtype=dtype)
    if forces is not None:
        data[AtomicDataDict.FORCE_KEY] = torch.as_tensor(np.asarray(forces), dtype=dtype)
    return compute_neighborlist_(data, r_max=r_max)


def graph_from_atoms(atoms: ase.Atoms, r_max: float = 5.0, dtype: torch.dtype = torch.float32):
    return make_graph(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc, r_max=r_max, dtype=dtype)


def collate(graphs: List[AtomicDataDict.Type]) -> AtomicDataDict.Type:
    return AtomicDataDict.batched_from_list(graphs)
