"""eIP: evidential deep learning for interatomic potentials, with its head on a NequIP backbone.

Reference: Xu et al., "Evidential Deep Learning for Interatomic Potentials", arXiv:2407.13994
(Nature Communications 2025), and its official code https://github.com/xuhan323/eIP
(commit 04190fb52b1ee67207ada6e0562ed8e469c8a6b1: ``PaiNN.py`` ``update_u``, ``run.py``
``quant_evi_loss``). The head and loss follow that code; the published backbone is PaiNN.

Here the backbone is the NequIP model of ``md22nop.models.e2ip`` (same as e2IP-NequIP). eIP reads
the Cartesian vector features of PaiNN's last layer; NequIP's last layer is scalar-only, so the l=1
(``1o``) features of the second-to-last layer are used (the same layer e2IP-NequIP reads). In
e3nn 0.6.0 the ``1o`` components are ordered (x, y, z), matching the force components.

Per atom and force component c the head outputs the Normal-Inverse-Gamma parameters
    nu_c = softplus(.) + 1e-5, alpha_c = softplus(.) + 1 + 1e-5, beta_c = softplus(.)
and the prediction is gamma = F = -dE/dx.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from nequip.data import AtomicDataDict
from nequip.nn import ForceStressOutput

from md22nop.models.e2ip import E2IP_FEATURES_KEY, _nequip_energy_model


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = math.log(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift


class EIPHead(torch.nn.Module):
    """``update_u`` of the official eIP code: Linear -> ShiftedSoftplus -> Linear(3), applied to each
    Cartesian component of the vector features, split into (alpha, beta, nu)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.act = ShiftedSoftplus()
        self.lin4 = torch.nn.Linear(hidden, 3)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin4.weight)
        self.lin4.bias.data.fill_(0)

    def forward(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        # v: [n_atoms, 3 (x, y, z), hidden]
        tmp = self.lin4(self.act(self.lin1(v)))
        f_alpha, f_beta, f_v = tmp.chunk(3, dim=-1)
        return {
            "eip_alpha": F.softplus(f_alpha.squeeze(-1)) + 1 + 10e-6,
            "eip_beta": F.softplus(f_beta.squeeze(-1)),
            "eip_nu": F.softplus(f_v.squeeze(-1)) + 10e-6,
        }


class EIPNequIP(torch.nn.Module):
    """NequIP energy model (forces = -grad E) plus the eIP evidential head."""

    def __init__(self, energy_model: ForceStressOutput, head: EIPHead, vec_slice: slice, vec_mul: int, config: Dict):
        super().__init__()
        self.energy_model = energy_model
        self.head = head
        self.vec_start, self.vec_stop = vec_slice.start, vec_slice.stop
        self.vec_mul = vec_mul
        self.config = config

    def vector_features(self, feats: torch.Tensor) -> torch.Tensor:
        # e3nn layout of "mul x 1o" is [mul, 3]; eIP's PaiNN layout is [3, hidden].
        return feats[:, self.vec_start:self.vec_stop].reshape(-1, self.vec_mul, 3).transpose(1, 2)

    def forward(self, data: AtomicDataDict.Type) -> Dict[str, torch.Tensor]:
        out = self.energy_model(dict(data))
        ev = self.head(self.vector_features(out[E2IP_FEATURES_KEY]))
        ev["energy"] = out[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
        ev["forces"] = out[AtomicDataDict.FORCE_KEY]
        if AtomicDataDict.STRESS_KEY in out:
            ev["stress"] = out[AtomicDataDict.STRESS_KEY]
        return ev


def build_eip_nequip(
    type_names: Sequence[str],
    r_max: float = 5.0,
    num_layers: int = 4,
    l_max: int = 2,
    parity: bool = False,
    num_features: Union[int, List[int]] = (128, 64, 32),
    type_embed_num_features: int = 32,
    radial_mlp_depth: int = 1,
    radial_mlp_width: int = 128,
    num_bessels: int = 8,
    polynomial_cutoff_p: int = 6,
    avg_num_neighbors: Optional[float] = None,
    per_type_energy_scales: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_shifts: Optional[Union[float, Sequence[float]]] = None,
) -> EIPNequIP:
    assert num_layers >= 2 and l_max >= 1, "eIP needs l=1 features from a non-final layer"
    config = {k: v for k, v in locals().items()}
    config["type_names"] = list(type_names)
    config["num_features"] = list(num_features) if not isinstance(num_features, int) else num_features
    config["model_type"] = "eip"
    energy_model, hidden, num_features = _nequip_energy_model(
        type_names, r_max, num_layers, l_max, parity, num_features, type_embed_num_features,
        radial_mlp_depth, radial_mlp_width, num_bessels, polynomial_cutoff_p, avg_num_neighbors,
        per_type_energy_scales, per_type_energy_shifts, save_features=True,
    )
    irreps = o3.Irreps(hidden)
    vec = None
    for mul_ir, sl in zip(irreps, irreps.slices()):
        if mul_ir.ir == o3.Irrep("1o"):
            vec = (sl, mul_ir.mul)
    assert vec is not None, f"no 1o features in {hidden}"
    return EIPNequIP(energy_model, EIPHead(vec[1]), vec[0], vec[1], config)


def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile: float = 0.6, coeff: float = 0.1):
    """Elementwise ``quant_evi_loss`` of the official eIP code (run.py), clamps included.

    NLL of Eq. (8) of the paper plus ``coeff`` times the regularizer. The code's regularizer is
    rho_q(y - gamma) * (2 nu + alpha + 1/beta) * (y - gamma), i.e. Eq. (9) with an extra (y - gamma)
    factor; the code is followed here.
    """
    alpha = torch.clamp(alpha, min=1 + 1e-6)
    beta = torch.clamp(beta, min=1e-6)
    v = torch.clamp(v, min=1e-6)
    z = beta / (alpha - 1)
    omega = 2 / (quantile * (1 - quantile))
    Omega = 4 * beta * (1 + omega * v * z)
    Omega = torch.clamp(Omega, min=1e-6, max=1e2)
    tau = (1 - 2 * quantile) / (quantile * (1 - quantile))
    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(Omega)
        + (alpha + 0.5) * torch.log(v * (y_true - gamma - tau * z) ** 2 + Omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    nll = torch.clamp(nll, min=1e-6, max=1e5)
    Psi = 2 * v + alpha + 1 / beta
    rho_q = torch.max(quantile * (y_true - gamma), (quantile - 1) * (y_true - gamma))
    reg = rho_q * Psi * (y_true - gamma)
    reg = torch.clamp(reg, min=1e-6, max=1e5)
    return nll + coeff * reg


def eip_epistemic(v, alpha, beta) -> torch.Tensor:
    """Per-atom uncertainty of Eqs. (5)-(6): per component beta / (nu (alpha - 1)), combined over x, y, z
    as sqrt(sum_c var_c^2)."""
    var = beta / (v * (alpha - 1))
    return torch.sqrt((var ** 2).sum(-1))
