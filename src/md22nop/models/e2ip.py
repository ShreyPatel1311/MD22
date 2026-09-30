"""e2IP: equivariant evidential deep learning for interatomic potentials, on a NequIP backbone.

Reference: Wang et al., "Equivariant Evidential Deep Learning for Interatomic Potentials",
arXiv:2602.10419 (Sec. 3, Algorithm 1, App. C-E).

Backbone modules are taken from NequIP (nequip==0.19.1, ``FullNequIPGNNModel``). The only change
to the backbone is a pass-through module that stores the node features of the second-to-last
interaction layer (which carry l=2 irreps; NequIP's last layer is scalar-only) so the evidential
covariance head can read them.

Per atom the model outputs
    gamma   : force mean, gamma = -dE/dx (Algorithm 1, line 3)
    nu      : covariance evidence, nu = softplus(.) + (d + 2)          (Eq. 9)
    kappa   : mean evidence,       kappa = softplus(.) + 1e-6           (Eq. 9)
    S       : symmetric 3x3 tangent vector from (0e + 2e) coefficients (Sec. 3.2)
    Sigma_0 : sigma_F**2 * expm(S), SPD and SO(3)-equivariant           (Eq. 10)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.io import CartesianTensor
from nequip.data import AtomicDataDict, register_fields
from nequip.model.energy_modules import _append_energy_modules
from nequip.nn import (
    ApplyFactor,
    ConvNetLayer,
    ForceStressOutput,
    PerTypeScaleShift,
    ScalarMLP,
    SequentialGraphNetwork,
)
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.nn.embedding import (
    BesselEdgeLengthEncoding,
    EdgeLengthNormalizer,
    NodeTypeEmbed,
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
)

E2IP_FEATURES_KEY = "e2ip_equivariant_features"
register_fields(node_fields=[E2IP_FEATURES_KEY])

D = 3  # force dimension


class SaveNodeFeatures(GraphModuleMixin, torch.nn.Module):
    """Copies the current node features into ``E2IP_FEATURES_KEY``."""

    def __init__(self, irreps_in):
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out={E2IP_FEATURES_KEY: irreps_in[AtomicDataDict.NODE_FEATURES_KEY]},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[E2IP_FEATURES_KEY] = data[AtomicDataDict.NODE_FEATURES_KEY]
        return data


def linear_tanh_damper(x: torch.Tensor, tau: float = 4.0, ceil: float = 5.0) -> torch.Tensor:
    """phi_high of Eq. (28): identity for |x| <= tau, smooth saturation to +-ceil beyond."""
    ax = x.abs()
    sat = torch.sign(x) * ((ceil - tau) * torch.tanh((ax - tau) / (ceil - tau)) + tau)
    return torch.where(ax <= tau, x, sat)


def _mlp(dims: Sequence[int]) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


class EvidentialHead(torch.nn.Module):
    """Evidential scalar head (nu, kappa) and equivariant covariance head (0e + 2e -> SPD).

    The l=2 channel is built only from equivariant linear maps on l=2 features and multiplication
    by invariant gates, so S transforms as R S R^T.
    """

    def __init__(
        self,
        irreps_equivariant: o3.Irreps,
        num_final_scalars: int,
        hidden: int = 32,
        force_scale: float = 0.1,
        damp_tau: float = 4.0,
        damp_ceil: float = 5.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.force_scale = force_scale
        self.damp_tau = damp_tau
        self.damp_ceil = damp_ceil
        self.lin0 = o3.Linear(irreps_equivariant, o3.Irreps(f"{hidden}x0e"))
        self.lin2 = o3.Linear(irreps_equivariant, o3.Irreps(f"{hidden}x2e"))
        n_inv = hidden + num_final_scalars
        self.gate = _mlp([n_inv, hidden, hidden])
        self.w_out = torch.nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.scalar_mlp = _mlp([n_inv, hidden, 1])
        self.evidence_mlp = _mlp([n_inv, hidden, 2])
        # (0e + 2e) -> symmetric Cartesian 3x3 via e3nn's fixed Clebsch-Gordan change of basis
        # (the paper's App. D uses e3nn.io.CartesianTensor for this map).
        self.cartesian = CartesianTensor("ij=ji")
        self.register_buffer("cob", self.cartesian.reduced_tensor_products().change_of_basis.clone())

    def forward(self, x_eq: torch.Tensor, x_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = x_eq.shape[0]
        h0 = self.lin0(x_eq)
        h2 = self.lin2(x_eq).view(n, self.hidden, 5)
        z = torch.cat([h0, x_final], dim=-1)
        gate = torch.sigmoid(self.gate(z))
        t = torch.einsum("nhm,nh,h->nm", h2, gate, self.w_out)
        s = self.scalar_mlp(z).squeeze(-1)
        ev = self.evidence_mlp(z)

        # Spectral stabilization by coefficient-space damping (Sec. 3.3, Eq. 12).
        # The damper is two-sided, so it bounds the scalar channel from above and below.
        s = linear_tanh_damper(s, self.damp_tau, self.damp_ceil)
        t_norm = t.norm(dim=-1, keepdim=True)
        t = t * linear_tanh_damper(t_norm, self.damp_tau, self.damp_ceil) / (t_norm + 1e-6)

        # sqrt(3) makes the isotropic part of S equal to s * I, so s is a log-variance directly.
        coeffs = torch.cat([math.sqrt(3.0) * s.unsqueeze(-1), t], dim=-1)
        S = torch.einsum("nk,kij->nij", coeffs, self.cob.to(coeffs.dtype))
        S = 0.5 * (S + S.transpose(-1, -2))

        nu = F.softplus(ev[:, 0]) + (D + 2)
        kappa = F.softplus(ev[:, 1]) + 1e-6
        return {"nu": nu, "kappa": kappa, "S": S}


class E2IPNequIP(torch.nn.Module):
    """NequIP energy model (forces = -grad E) plus the e2IP evidential head."""

    def __init__(self, energy_model: ForceStressOutput, head: EvidentialHead, config: Dict):
        super().__init__()
        self.energy_model = energy_model
        self.head = head
        self.config = config

    def forward(self, data: AtomicDataDict.Type) -> Dict[str, torch.Tensor]:
        # NequIP writes predictions into the dict it is given; a shallow copy keeps the caller's
        # reference labels (same keys) intact.
        out = self.energy_model(dict(data))
        ev = self.head(out[E2IP_FEATURES_KEY], out[AtomicDataDict.NODE_FEATURES_KEY])
        ev["energy"] = out[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
        ev["forces"] = out[AtomicDataDict.FORCE_KEY]
        if AtomicDataDict.STRESS_KEY in out:
            ev["stress"] = out[AtomicDataDict.STRESS_KEY]
        return ev

    def sigma0(self, S: torch.Tensor) -> torch.Tensor:
        return self.head.force_scale**2 * torch.linalg.matrix_exp(S)


def _nequip_energy_model(
    type_names: Sequence[str],
    r_max: float,
    num_layers: int,
    l_max: int,
    parity: bool,
    num_features: Union[int, List[int]],
    type_embed_num_features: int,
    radial_mlp_depth: int,
    radial_mlp_width: int,
    num_bessels: int,
    polynomial_cutoff_p: int,
    avg_num_neighbors: Optional[float],
    per_type_energy_scales,
    per_type_energy_shifts,
    save_features: bool,
):
    """Mirrors ``nequip.model.nequip_models.FullNequIPGNNModel`` (nequip 0.19.1) with the default
    ``convnet_*`` settings. With ``save_features`` a ``SaveNodeFeatures`` pass-through is inserted
    before the final scalar-only layer. Returns (energy model, hidden irreps, num_features list)."""
    if isinstance(num_features, int):
        num_features = [num_features] * (l_max + 1)
    num_features = list(num_features)
    irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(lmax=l_max))
    hidden = repr(
        o3.Irreps(
            [
                (num_features[l], (l, p))
                for l in range(l_max + 1)
                for p in ((1, -1) if parity else ((1,) if l % 2 == 0 else (-1,)))
            ]
        )
    )
    feature_irreps = [hidden] * (num_layers - 1) + [repr(o3.Irreps([(num_features[0], (0, 1))]))]

    type_embed = NodeTypeEmbed(type_names=type_names, num_features=type_embed_num_features)
    spharm = SphericalHarmonicEdgeAttrs(irreps_edge_sh=irreps_edge_sh, irreps_in=type_embed.irreps_out)
    edge_norm = EdgeLengthNormalizer(r_max=r_max, type_names=type_names, irreps_in=spharm.irreps_out)
    bessel = BesselEdgeLengthEncoding(
        num_bessels=num_bessels,
        trainable=False,
        cutoff=PolynomialCutoff(polynomial_cutoff_p),
        edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )
    factor = ApplyFactor(
        in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        factor=(2 * math.pi) / (r_max * r_max),
        irreps_in=bessel.irreps_out,
    )
    modules = {
        "type_embed": type_embed,
        "spharm": spharm,
        "edge_norm": edge_norm,
        "bessel_encode": bessel,
        "factor": factor,
    }
    prev = factor.irreps_out
    for i in range(num_layers):
        layer = ConvNetLayer(
            irreps_in=prev,
            feature_irreps_hidden=feature_irreps[i],
            convolution_kwargs={
                "radial_mlp_depth": radial_mlp_depth,
                "radial_mlp_width": radial_mlp_width,
                "use_sc": i != 0,
                "is_first_layer": i == 0,
                "avg_num_neighbors": avg_num_neighbors,
                "type_names": type_names,
            },
            resnet=False,
            nonlinearity_type="gate",
            nonlinearity_scalars={"e": "silu", "o": "tanh"},
            nonlinearity_gates={"e": "silu", "o": "tanh"},
        )
        prev = layer.irreps_out
        modules[f"layer{i}_convnet"] = layer
        if save_features and i == num_layers - 2:
            save = SaveNodeFeatures(irreps_in=prev)
            prev = save.irreps_out
            modules["save_e2ip_features"] = save

    readout = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=0,
        hidden_layers_width=o3.Irreps(feature_irreps[-1]).dim,
        nonlinearity="silu",
        bias=False,
        forward_weight_init=True,
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        irreps_in=prev,
    )
    scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        scales=per_type_energy_scales,
        shifts=per_type_energy_shifts,
        scales_trainable=False,
        shifts_trainable=False,
        irreps_in=readout.irreps_out,
    )
    modules["per_atom_energy_readout"] = readout
    modules["per_type_energy_scale_shift"] = scale_shift
    energy_model = _append_energy_modules(SequentialGraphNetwork(modules), type_names=type_names)
    energy_model = ForceStressOutput(energy_model, do_derivatives=True)

    return energy_model, hidden, num_features


def build_e2ip_nequip(
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
    head_hidden: int = 32,
    force_scale: float = 0.1,
) -> E2IPNequIP:
    """NequIP backbone (see ``_nequip_energy_model``) plus the e2IP evidential head."""
    assert num_layers >= 2 and l_max >= 2, "e2IP needs l=2 features from a non-final layer"
    config = {k: v for k, v in locals().items()}
    config["type_names"] = list(type_names)
    config["num_features"] = list(num_features) if not isinstance(num_features, int) else num_features
    config["model_type"] = "e2ip"

    energy_model, hidden, num_features = _nequip_energy_model(
        type_names, r_max, num_layers, l_max, parity, num_features, type_embed_num_features,
        radial_mlp_depth, radial_mlp_width, num_bessels, polynomial_cutoff_p, avg_num_neighbors,
        per_type_energy_scales, per_type_energy_shifts, save_features=True,
    )
    head = EvidentialHead(
        irreps_equivariant=o3.Irreps(hidden),
        num_final_scalars=num_features[0],
        hidden=head_hidden,
        force_scale=force_scale,
    )
    return E2IPNequIP(energy_model, head, config)


class PlainNequIP(torch.nn.Module):
    """Baseline without the e2IP head: the same NequIP energy model, forces = -grad E."""

    def __init__(self, energy_model: ForceStressOutput, config: Dict):
        super().__init__()
        self.energy_model = energy_model
        self.config = config

    def forward(self, data: AtomicDataDict.Type) -> Dict[str, torch.Tensor]:
        out = self.energy_model(dict(data))
        ev = {"energy": out[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1), "forces": out[AtomicDataDict.FORCE_KEY]}
        if AtomicDataDict.STRESS_KEY in out:
            ev["stress"] = out[AtomicDataDict.STRESS_KEY]
        return ev


def build_plain_nequip(
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
) -> PlainNequIP:
    """The e2IP backbone without ``SaveNodeFeatures`` and without the evidential head."""
    config = {k: v for k, v in locals().items()}
    config["type_names"] = list(type_names)
    config["num_features"] = list(num_features) if not isinstance(num_features, int) else num_features
    config["model_type"] = "nequip"
    energy_model, _, _ = _nequip_energy_model(
        type_names, r_max, num_layers, l_max, parity, num_features, type_embed_num_features,
        radial_mlp_depth, radial_mlp_width, num_bessels, polynomial_cutoff_p, avg_num_neighbors,
        per_type_energy_scales, per_type_energy_shifts, save_features=False,
    )
    return PlainNequIP(energy_model, config)


def e2ip_nll(y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor, kappa: torch.Tensor,
             S: torch.Tensor, force_scale: float) -> torch.Tensor:
    """Per-atom multivariate Student-t NLL, Eq. (13)/(25), with Sigma_0 = force_scale^2 expm(S).

    log|Sigma_0| = tr(S) + 2d log(force_scale) and Sigma_0^{-1} = expm(-S) / force_scale^2, so no
    Cholesky factorization is needed.
    """
    v = y - gamma
    inv = torch.linalg.matrix_exp(-S) / force_scale**2
    M = torch.einsum("ni,nij,nj->n", v, inv, v)
    logdet = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) + 2 * D * math.log(force_scale)
    return (
        torch.lgamma((nu - D + 1) / 2)
        - torch.lgamma((nu + 1) / 2)
        + 0.5 * D * torch.log(math.pi * nu * (1 + kappa) / kappa)
        + 0.5 * logdet
        + 0.5 * (nu + 1) * torch.log1p(kappa / (nu * (1 + kappa)) * M)
    )


def e2ip_regularizer(y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Evidence regularizer, Eq. (14): (nu + kappa) * ||y - gamma||^2."""
    return (nu + kappa) * (y - gamma).pow(2).sum(-1)


def uncertainty_tensors(model: E2IPNequIP, nu, kappa, S):
    """Aleatoric E[Sigma] and epistemic Var[mu] tensors, Eq. (8), and the scalar proxy of Eq. (15)."""
    sigma0 = model.sigma0(S)
    u_ale = (nu / (nu - D - 1))[:, None, None] * sigma0
    u_epi = u_ale / kappa[:, None, None]
    u_scalar = torch.sqrt(torch.diagonal(u_epi, dim1=-2, dim2=-1).sum(-1) / 3)
    return u_ale, u_epi, u_scalar
