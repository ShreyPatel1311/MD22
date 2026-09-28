#!/usr/bin/env bash
# End-to-end run on a Vast.ai GPU instance: data -> e2IP training -> FPBench (4 components).
# Requires HF_TOKEN in the instance environment (Vast account-level environment variable).
# Progress is pushed to the Hugging Face model repo under logs/ so it can be followed remotely.
set -euo pipefail

TRAIN_HOURS="${TRAIN_HOURS:-5}"
WORKERS="${WORKERS:-12}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Godseye1311/e2ip-nequip-matpes10}"
HF_DATA_REPO="${HF_DATA_REPO:-Godseye1311/matpes-pbe-10pct}"
REPO_URL="${REPO_URL:-https://github.com/ShreyPatel1311/MD22}"
REPO_REF="${REPO_REF:-claude/gallant-volta-9wfu97}"
FPBENCH_COMMIT="b8475daeab17b76404b0cde1cbe53ca1a60a92e5"
W=/workspace/run
mkdir -p "$W/logs" && cd "$W"
exec > >(tee -a "$W/logs/pipeline.log") 2>&1

stage() { echo "=== $(date -u +%FT%TZ) STAGE $*"; echo "$*" > "$W/logs/STAGE"; }
hf_up() { python -c "
import sys; from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=sys.argv[1], repo_type=sys.argv[2], folder_path=sys.argv[3], path_in_repo=sys.argv[4], commit_message=sys.argv[5])
" "$@"; }
log_sync() { while true; do hf_up "$HF_MODEL_REPO" model "$W/logs" logs "log sync" >/dev/null 2>&1 || true; sleep 600; done; }

stage setup
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set"; exit 1; }
nvidia-smi
pip install -q "nequip==0.19.1" "matcalc==0.5.1" pymatgen ijson ase-db-backends huggingface_hub pandas matplotlib scipy pytest
python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_MODEL_REPO", repo_type="model", private=True, exist_ok=True)
api.create_repo("$HF_DATA_REPO", repo_type="dataset", private=True, exist_ok=True)
EOF
log_sync & SYNC_PID=$!
[ -d MD22 ] || git clone -q -b "$REPO_REF" "$REPO_URL" MD22
[ -d FPBench ] || { git clone -q https://github.com/mogroupumd/FPBench FPBench && git -C FPBench checkout -q "$FPBENCH_COMMIT"; }
export PYTHONPATH="$W/MD22/src"
( cd MD22 && python -m pytest -q tests/ )

stage data-matpes
python MD22/scripts/prepare_data.py matpes --out data
mkdir -p hf_data && cp data/matpes_pbe_10pct_train.npz data/matpes_pbe_10pct_val.npz data/matpes_pbe_10pct_split.json hf_data/
hf_up "$HF_DATA_REPO" dataset hf_data . "MatPES-PBE v2025.1 10% subset (seed 0), 95/5 train/val"

stage train
python -m md22nop.training.train --train data/matpes_pbe_10pct_train.npz --val data/matpes_pbe_10pct_val.npz \
  --out ckpt --hours "$TRAIN_HOURS" --workers "$WORKERS" 2>&1 | tee logs/train.log
hf_up "$HF_MODEL_REPO" model ckpt checkpoints "e2IP-NequIP checkpoints"
CKPT="$W/ckpt/final.pt"

stage eval-force-matpes
python MD22/scripts/eval_force.py --checkpoint "$CKPT" --dataset data/matpes_pbe_full.npz --name matpes_pbe \
  --exclude-ids data/matpes_pbe_10pct_split.json --fpbench FPBench --out results/force --workers "$WORKERS" 2>&1 | tee logs/eval_force_matpes.log
rm -f results/force/matpes_pbe_force_results_standardized.json.gz; gzip -f results/force/matpes_pbe_force_results_standardized.json
hf_up "$HF_MODEL_REPO" model results results "FPBench force (MatPES-PBE)"

stage eval-phase
python MD22/scripts/eval_phase.py --checkpoint "$CKPT" --fpbench FPBench --out results/phase --workers "$WORKERS" 2>&1 | tee logs/eval_phase.log
hf_up "$HF_MODEL_REPO" model results results "FPBench phase stability & ordering"

stage eval-neb
python MD22/scripts/eval_neb.py --checkpoint "$CKPT" --fpbench FPBench --src MD22/src --out results/neb --workers "$WORKERS" 2>&1 | tee logs/eval_neb.log
rm -rf results/neb/neb_component/generation/runs
hf_up "$HF_MODEL_REPO" model results results "FPBench ion-migration NEB"

stage data-omat24
python MD22/scripts/prepare_data.py omat24 --out data

stage eval-force-omat24
python MD22/scripts/eval_force.py --checkpoint "$CKPT" --dataset data/omat24_rattled_1000.npz --name omat24_rattled_1000 \
  --fpbench FPBench --out results/force --workers "$WORKERS" 2>&1 | tee logs/eval_force_omat24.log
gzip -f results/force/omat24_rattled_1000_force_results_standardized.json
hf_up "$HF_MODEL_REPO" model results results "FPBench force (OMat24 rattled-1000)"

stage done
kill "$SYNC_PID" || true
hf_up "$HF_MODEL_REPO" model logs logs "final logs"
