"""ASE calculator for a trained e2IP-NequIP or plain NequIP checkpoint.

For e2IP checkpoints it also exposes the per-atom evidential outputs
(``e2ip_nu``, ``e2ip_kappa``, ``e2ip_epistemic_scalar``) in ``results``.
"""

from __future__ import annotations

import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from md22nop.data.graph import collate, graph_from_atoms
from md22nop.models.e2ip import uncertainty_tensors
from md22nop.training.train import load_model


class E2IPCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path: str, device: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = device
        self.model, ckpt = load_model(model_path, device)
        self.r_max = float(ckpt["model_config"]["r_max"])

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        data = collate([graph_from_atoms(self.atoms, r_max=self.r_max)])
        data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in data.items()}
        out = self.model(data)
        energy = float(out["energy"].detach().double().item())
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": out["forces"].detach().double().cpu().numpy(),
            # NequIP's stress follows the ASE sign convention (nequip.integrations.ase uses it as is).
            "stress": full_3x3_to_voigt_6_stress(out["stress"].detach().double().cpu().numpy().reshape(3, 3)),
        }
        if "nu" in out:
            _, _, u_scalar = uncertainty_tensors(self.model, out["nu"], out["kappa"], out["S"])
            self.results.update({
                "e2ip_nu": out["nu"].detach().cpu().numpy(),
                "e2ip_kappa": out["kappa"].detach().cpu().numpy(),
                "e2ip_epistemic_scalar": u_scalar.detach().cpu().numpy(),
            })
