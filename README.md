# MD22

**md22nop** — training neural operators for **trajectory prediction** on the
[MD22](http://www.sgdml.org/#datasets) benchmark.

The task is framed in the EGNO / ATOM setting: given an initial atomic
configuration, predict future positions along the molecular-dynamics
trajectory — not force-field energy/force regression.

## Repository layout

```
src/md22nop/        # Python package
  data/             # MD22 dataset loading / trajectory windowing
  models/           # architectures (registered in models/__init__.py::REGISTRY)
  training/         # plain-PyTorch Trainer
  hub/              # Hugging Face Hub integration (PyTorchModelHubMixin)
  utils/
configs/            # Hydra configs (data/, model/, trainer/)
scripts/            # entry-point scripts
tests/              # test suite
docs/               # notes
```

## Where things live

| Artifact | Location |
| --- | --- |
| Code | this GitHub repo (`src/` layout, package `md22nop`) |
| Data | Hugging Face **dataset** repo `md22-trajectories` (`raw/md22_*.npz` mirror of sGDML) |
| Weights | Hugging Face **model** repos `md22-<arch>-<molecule>` via `PyTorchModelHubMixin` |

Each Hugging Face model revision corresponds to one git tag, with the git SHA
recorded in the model card.

## Adding an architecture

1. Add a module under `src/md22nop/models/`.
2. Register it in `src/md22nop/models/__init__.py::REGISTRY`.
3. Add `configs/model/<arch>.yaml` with `name: <arch>`.

An `MLPBaseline` serves as the end-to-end smoke test.

## Reference

ATOM: *A Pretrained Neural Operator for Multitask Molecular Dynamics* (ICLR 2026).

## License

[MIT](LICENSE)
