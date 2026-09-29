import numpy as np
import pytest
import torch
from ase.build import bulk
from e3nn import o3

from md22nop.data.graph import TYPE_NAMES, collate, graph_from_atoms
from md22nop.models.e2ip import build_e2ip_nequip, e2ip_nll, e2ip_regularizer


@pytest.fixture(scope="module", autouse=True)
def float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(scope="module")
def model(float64):
    torch.manual_seed(0)
    m = build_e2ip_nequip(
        TYPE_NAMES, num_layers=3, num_features=[16, 8, 8], radial_mlp_width=16,
        avg_num_neighbors=40.0, per_type_energy_scales=1.0, per_type_energy_shifts=0.0,
        head_hidden=8,
    ).double()
    return m.eval()


def _atoms(seed=1):
    at = bulk("NaCl", "rocksalt", a=5.6).repeat((2, 1, 1))
    at.rattle(0.15, seed=seed)
    return at


def _run(model, atoms):
    g = graph_from_atoms(atoms, dtype=torch.float64)
    return model(collate([g]))


def test_forces_are_negative_energy_gradient(model):
    at = _atoms()
    f = _run(model, at)["forces"].detach().numpy()
    h, i, k = 1e-4, 3, 1
    ep, em = at.copy(), at.copy()
    ep.positions[i, k] += h
    em.positions[i, k] -= h
    fd = -(_run(model, ep)["energy"].item() - _run(model, em)["energy"].item()) / (2 * h)
    assert abs(fd - f[i, k]) < 1e-6 * max(1.0, abs(fd))


def test_covariance_is_spd_and_equivariant(model):
    at = _atoms()
    R = o3.rand_matrix(dtype=torch.float64).numpy()
    at2 = at.copy()
    at2.set_cell(at.cell.array @ R.T, scale_atoms=False)
    at2.positions = at.positions @ R.T
    o1, o2 = _run(model, at), _run(model, at2)
    Rt = torch.as_tensor(R)
    s1 = model.sigma0(o1["S"]).detach()
    s2 = model.sigma0(o2["S"]).detach()
    assert torch.linalg.eigvalsh(s1).min() > 0
    assert torch.allclose(s2, Rt @ s1 @ Rt.T, atol=1e-10, rtol=1e-8)
    assert torch.allclose(o2["forces"], o1["forces"] @ Rt.T, atol=1e-10)
    assert torch.allclose(o2["nu"], o1["nu"]) and torch.allclose(o2["kappa"], o1["kappa"])
    assert torch.allclose(o2["energy"], o1["energy"])


def test_loss_backward(model):
    model.train()
    g = collate([graph_from_atoms(_atoms(s), dtype=torch.float64) for s in (1, 2)])
    out = model(g)
    y = torch.randn_like(out["forces"])
    loss = (e2ip_nll(y, out["forces"], out["nu"], out["kappa"], out["S"], model.head.force_scale)
            + 0.1 * e2ip_regularizer(y, out["forces"], out["nu"], out["kappa"])).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(gr).all() for gr in grads)
    model.eval()


def test_nll_matches_scipy_student_t():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    S = torch.tensor((A + A.T) / 4)[None]
    nu, kappa, fs = torch.tensor([7.3]), torch.tensor([0.8]), 0.2
    y, gamma = torch.tensor(rng.normal(size=(1, 3))), torch.zeros(1, 3, dtype=torch.float64)
    nll = e2ip_nll(y, gamma, nu, kappa, S, fs).item()
    sigma0 = (fs**2 * torch.linalg.matrix_exp(S[0])).numpy()
    m = nu.item() - 3 + 1
    scale = nu.item() * (kappa.item() + 1) / (kappa.item() * m) * sigma0  # Eq. (7)
    ref = -stats.multivariate_t(loc=np.zeros(3), shape=scale, df=m).logpdf(y.numpy()[0])
    assert abs(nll - ref) < 1e-8


def test_forward_does_not_overwrite_input_labels(model):
    from nequip.data import AtomicDataDict
    from md22nop.data.graph import make_graph
    at = _atoms()
    e_ref, f_ref = -123.0, np.full((len(at), 3), 0.5)
    g = collate([make_graph(at.numbers, at.positions, at.cell.array, energy=e_ref, forces=f_ref,
                            dtype=torch.float64)])
    out = model(g)
    assert g[AtomicDataDict.TOTAL_ENERGY_KEY].item() == e_ref
    assert torch.equal(g[AtomicDataDict.FORCE_KEY], torch.as_tensor(f_ref))
    assert not torch.allclose(out["forces"], g[AtomicDataDict.FORCE_KEY])


def test_plain_nequip_forces_and_checkpoint_roundtrip(tmp_path):
    from md22nop.models.e2ip import build_plain_nequip
    from md22nop.training.train import load_model

    torch.manual_seed(0)
    kw = dict(num_layers=3, num_features=[16, 8, 8], radial_mlp_width=16, avg_num_neighbors=40.0,
              per_type_energy_scales=1.0, per_type_energy_shifts=0.0)
    m = build_plain_nequip(TYPE_NAMES, **kw).double().eval()
    out = _run(m, _atoms())
    assert set(out) >= {"energy", "forces"} and "nu" not in out
    torch.save({"model_config": m.config, "state_dict": m.state_dict()}, tmp_path / "p.pt")
    m2, _ = load_model(tmp_path / "p.pt")
    out2 = _run(m2.double(), _atoms())
    assert torch.allclose(out["forces"], out2["forces"])


def test_eip_vector_features_rotate_and_loss_backward(tmp_path):
    from e3nn import o3
    from md22nop.models.eip import build_eip_nequip, quant_evi_loss
    from md22nop.models.e2ip import E2IP_FEATURES_KEY
    from md22nop.training.train import load_model

    torch.manual_seed(0)
    m = build_eip_nequip(TYPE_NAMES, num_layers=3, num_features=[16, 8, 8], radial_mlp_width=16,
                         avg_num_neighbors=40.0, per_type_energy_scales=1.0, per_type_energy_shifts=0.0).double().eval()
    at = _atoms()
    R = o3.rand_matrix().double()
    at_r = at.copy()
    at_r.positions = at.positions @ R.numpy().T
    at_r.cell = at.cell.array @ R.numpy().T

    def feats(a):
        out = m.energy_model(dict(collate([graph_from_atoms(a, dtype=torch.float64)])))
        return m.vector_features(out[E2IP_FEATURES_KEY]).detach()

    v, v_r = feats(at), feats(at_r)  # [n, 3, mul]
    assert torch.allclose(torch.einsum("ij,njm->nim", R, v), v_r, atol=1e-8)

    out = _run(m, at)
    for k in ("eip_nu", "eip_alpha", "eip_beta"):
        assert out[k].shape == out["forces"].shape
    m.train()
    out_t = _run(m, at)
    y = out_t["forces"].detach() + 0.1
    loss = quant_evi_loss(y, out_t["forces"], out_t["eip_nu"], out_t["eip_alpha"], out_t["eip_beta"]).mean(0).sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
    m.eval()

    torch.save({"model_config": m.config, "state_dict": m.state_dict()}, tmp_path / "e.pt")
    m2, _ = load_model(tmp_path / "e.pt")
    assert torch.allclose(_run(m2.double(), at)["eip_nu"], _run(m, at)["eip_nu"])
