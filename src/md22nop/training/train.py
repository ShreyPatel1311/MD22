"""Time-budgeted e2IP training on MatPES-PBE subsets.

Loss (per batch):
    L = lambda_E * mean_structures(((E_pred - E) / N_atoms)^2)      [eV^2 / atom^2]
      + lambda_F * mean_atoms(NLL_i + lambda_reg * reg_i)
NLL_i and reg_i are Eqs. (13)-(14) of the e2IP paper; lambda_reg = 0.1 follows its Table 5.
The paper does not state its energy loss, so the energy term follows mace-torch's defaults
(per-atom energy MSE, energy weight 1.0, raised to 1000.0 for the last quarter of training,
``--energy_weight`` / ``--stage_two_energy_weight`` / ``start_swa = 3/4 of epochs``).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from md22nop.data.graph import TYPE_NAMES, collate, from_numpy, make_graph, to_numpy
from md22nop.data.matpes import iter_structures
from md22nop.models.e2ip import build_e2ip_nequip, e2ip_nll, e2ip_regularizer
from nequip.data import AtomicDataDict

R_MAX = 5.0


def _graph_worker(args):
    i, z, pos, cell, e, f = args
    return to_numpy(make_graph(z, pos, cell, r_max=R_MAX, energy=e, forces=f))


def build_graphs(arrays: Dict[str, np.ndarray], workers: int) -> List[AtomicDataDict.Type]:
    items = list(iter_structures(arrays))
    with mp.Pool(workers) as pool:
        return [from_numpy(g) for g in pool.imap(_graph_worker, items, chunksize=64)]


def dataset_stats(graphs, n_types: int = len(TYPE_NAMES)):
    counts = np.zeros((len(graphs), n_types))
    energies = np.zeros(len(graphs))
    f2, n_atoms, n_edges = 0.0, 0, 0
    for k, g in enumerate(graphs):
        t = g[AtomicDataDict.ATOM_TYPE_KEY].numpy()
        np.add.at(counts[k], t, 1)
        energies[k] = g[AtomicDataDict.TOTAL_ENERGY_KEY].item()
        f2 += g[AtomicDataDict.FORCE_KEY].double().pow(2).sum().item()
        n_atoms += len(t)
        n_edges += g[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    seen = counts.sum(0) > 0
    sol, *_ = np.linalg.lstsq(counts[:, seen], energies, rcond=None)
    shifts = np.full(n_types, float(np.median(sol)))
    shifts[seen] = sol
    return {
        "per_type_energy_shifts": {TYPE_NAMES[i]: float(shifts[i]) for i in range(n_types)},
        "force_rms": math.sqrt(f2 / (3 * n_atoms)),
        "avg_num_neighbors": n_edges / n_atoms,
        "seen_types": [TYPE_NAMES[i] for i in np.flatnonzero(seen)],
        "n_structures": len(graphs),
        "n_atoms": n_atoms,
    }


def make_batches(graphs, max_atoms: int, rng: np.random.Generator, shuffle: bool = True):
    order = rng.permutation(len(graphs)) if shuffle else np.arange(len(graphs))
    batch, n = [], 0
    for i in order:
        na = graphs[i][AtomicDataDict.POSITIONS_KEY].shape[0]
        if batch and n + na > max_atoms:
            yield batch
            batch, n = [], 0
        batch.append(i)
        n += na
    if batch:
        yield batch


def to_device(data, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}


def compute_loss(model, out, batch, cfg, lambda_e):
    n = batch[AtomicDataDict.NUM_NODES_KEY].to(out["energy"].dtype)
    e_ref = batch[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
    loss_e = (((out["energy"] - e_ref) / n) ** 2).mean()
    y = batch[AtomicDataDict.FORCE_KEY]
    nll = e2ip_nll(y, out["forces"], out["nu"], out["kappa"], out["S"], model.head.force_scale)
    reg = e2ip_regularizer(y, out["forces"], out["nu"], out["kappa"])
    loss_f = (nll + cfg["lambda_reg"] * reg).mean()
    loss = lambda_e * loss_e + cfg["lambda_f"] * loss_f
    return loss, {"loss_e": loss_e.item(), "loss_f": loss_f.item(), "nll": nll.mean().item()}


@torch.no_grad()
def _ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def evaluate(model, graphs, device, max_atoms):
    model.eval()
    e_abs, f_abs, nll_sum, n_struct, n_atoms = 0.0, 0.0, 0.0, 0, 0
    for idx in make_batches(graphs, max_atoms, np.random.default_rng(0), shuffle=False):
        batch = to_device(collate([graphs[i] for i in idx]), device)
        out = model(batch)
        n = batch[AtomicDataDict.NUM_NODES_KEY].to(out["energy"].dtype)
        e_ref = batch[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1)
        y = batch[AtomicDataDict.FORCE_KEY]
        with torch.no_grad():
            e_abs += ((out["energy"] - e_ref) / n).abs().sum().item()
            f_abs += (out["forces"] - y).abs().sum().item()
            nll_sum += e2ip_nll(y, out["forces"], out["nu"], out["kappa"], out["S"], model.head.force_scale).sum().item()
        n_struct += len(idx)
        n_atoms += y.shape[0]
    return {
        "energy_mae_meV_per_atom": 1000 * e_abs / n_struct,
        "force_mae_meV_per_A": 1000 * f_abs / (3 * n_atoms),
        "force_nll_per_atom": nll_sum / n_atoms,
    }


def save_checkpoint(path: Path, model, stats, cfg, history):
    torch.save({"model_config": model.config, "state_dict": model.state_dict(), "stats": stats,
                "train_config": cfg, "history": history}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    model = build_e2ip_nequip(**{k: v for k, v in cfg.items() if k != "config"})
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--max-atoms", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--val-every-min", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_g = build_graphs(dict(np.load(args.train)), args.workers)
    val_g = build_graphs(dict(np.load(args.val)), args.workers)
    stats = dataset_stats(train_g)
    print(f"graphs built in {time.time() - t0:.0f}s: train={len(train_g)} val={len(val_g)} stats="
          f"{ {k: v for k, v in stats.items() if k != 'per_type_energy_shifts'} }", flush=True)

    # NequIP "M" preset (nequip 0.19.1 _NEQUIP_GNN_PRESETS / _NEQUIP_GNN_STANDARD_PRESET).
    model_kwargs = dict(
        type_names=TYPE_NAMES, r_max=R_MAX, num_layers=4, l_max=2, parity=False, num_features=[128, 64, 32],
        type_embed_num_features=32, radial_mlp_depth=1, radial_mlp_width=128,
        avg_num_neighbors=stats["avg_num_neighbors"], per_type_energy_scales=stats["force_rms"],
        per_type_energy_shifts=stats["per_type_energy_shifts"], head_hidden=32, force_scale=0.1,
    )
    model = build_e2ip_nequip(**model_kwargs).to(device)
    ema = build_e2ip_nequip(**model_kwargs).to(device).eval()
    ema.load_state_dict(model.state_dict())
    for q in ema.parameters():
        q.requires_grad_(False)
    n_params = sum(q.numel() for q in model.parameters())
    cfg = {"lambda_e": 1.0, "lambda_e_stage_two": 1000.0, "stage_two_start_frac": 0.75, "lambda_f": 1.0, "lambda_reg": 0.1, "lr": args.lr,
           "weight_decay": 1e-3, "ema_decay": 0.999, "grad_clip": 10.0, "max_atoms": args.max_atoms,
           "hours": args.hours, "n_params": n_params, "r_max": R_MAX}
    print(f"model params: {n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["weight_decay"])

    budget = args.hours * 3600
    t_start, last_val, step, epoch = time.time(), time.time(), 0, 0
    history, best = [], float("inf")
    rng = np.random.default_rng(args.seed)
    done = False
    while not done:
        epoch += 1
        for idx in make_batches(train_g, args.max_atoms, rng):
            frac = (time.time() - t_start) / budget
            if frac >= 1.0:
                done = True
                break
            lr = args.lr * (frac / args.warmup_frac if frac < args.warmup_frac else
                            0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (frac - args.warmup_frac) / (1 - args.warmup_frac))))
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            model.train()
            batch = to_device(collate([train_g[i] for i in idx]), device)
            out_ = model(batch)
            lambda_e = cfg["lambda_e_stage_two"] if frac >= cfg["stage_two_start_frac"] else cfg["lambda_e"]
            loss, parts = compute_loss(model, out_, batch, cfg, lambda_e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if torch.isfinite(loss) and torch.isfinite(gnorm):
                opt.step()
                _ema_update(ema, model, cfg["ema_decay"])
            step += 1
            if step % 100 == 0:
                el = time.time() - t_start
                print(f"step {step} ep {epoch} t={el/60:.1f}min lr={lr:.2e} loss={loss.item():.4f} "
                      f"{json.dumps({k: round(v, 5) for k, v in parts.items()})} gnorm={gnorm:.2f}", flush=True)
            if time.time() - last_val > args.val_every_min * 60:
                last_val = time.time()
                metrics = evaluate(ema, val_g, device, args.max_atoms)
                metrics.update({"step": step, "epoch": epoch, "minutes": (time.time() - t_start) / 60})
                history.append(metrics)
                print("VAL", json.dumps(metrics), flush=True)
                save_checkpoint(out / "last.pt", ema, stats, cfg, history)
                score = metrics["force_mae_meV_per_A"] + 10 * metrics["energy_mae_meV_per_atom"]
                if score < best:
                    best = score
                    save_checkpoint(out / "best.pt", ema, stats, cfg, history)

    metrics = evaluate(ema, val_g, device, args.max_atoms)
    metrics.update({"step": step, "epoch": epoch, "minutes": (time.time() - t_start) / 60, "final": True})
    history.append(metrics)
    print("VAL", json.dumps(metrics), flush=True)
    save_checkpoint(out / "final.pt", ema, stats, cfg, history)
    (out / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
