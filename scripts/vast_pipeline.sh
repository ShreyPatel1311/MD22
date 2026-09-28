#!/usr/bin/env bash
# End-to-end run on a Vast.ai GPU instance: data -> e2IP training -> FPBench (4 components).
# With HF_TOKEN in the environment, data, checkpoints, results and logs are pushed to Hugging Face.
# Without it every artifact stays under /workspace/run and the upload steps are skipped.
# With CKPT_IN set, training is skipped: the script waits (up to 3 h) for that checkpoint file,
# e.g. copied in from a training instance with Vast's copy_direct, and runs the evaluations on it.
# Evaluation stages are independent: a failed stage is logged as FAILED and the next one still runs.
set -euo pipefail

TRAIN_HOURS="${TRAIN_HOURS:-5}"
EVAL_STAGES="${EVAL_STAGES:-force-matpes phase neb omat24}"   # evaluation stages to run
OMAT_FRACTION="${OMAT_FRACTION:-0.1}"                            # random fraction of OMat24 rattled-1000
NEB_FRACTION="${NEB_FRACTION:-0.1}"                              # random fraction of the 154 NEB pathways
want() { case " $EVAL_STAGES " in *" $1 "*) return 0;; esac; return 1; }
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
hf_up() { [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN unset: skipping upload of $3"; return 0; }; python -c "
import sys; from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=sys.argv[1], repo_type=sys.argv[2], folder_path=sys.argv[3], path_in_repo=sys.argv[4], commit_message=sys.argv[5])
" "$@"; }
log_sync() { while true; do hf_up "$HF_MODEL_REPO" model "$W/logs" logs "log sync" >/dev/null 2>&1 || true; sleep 600; done; }

stage setup
nvidia-smi
pip install -q "nequip==0.19.1" "matcalc==0.5.1" pymatgen ijson ase-db-backends huggingface_hub pandas matplotlib seaborn scipy pytest
if [ -n "${HF_TOKEN:-}" ]; then python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$HF_MODEL_REPO", repo_type="model", private=True, exist_ok=True)
api.create_repo("$HF_DATA_REPO", repo_type="dataset", private=True, exist_ok=True)
EOF
fi
SYNC_PID=""
[ -z "${HF_TOKEN:-}" ] || { log_sync & SYNC_PID=$!; }
[ -d MD22 ] || git clone -q -b "$REPO_REF" "$REPO_URL" MD22
[ -d FPBench ] || { git clone -q https://github.com/mogroupumd/FPBench FPBench && git -C FPBench checkout -q "$FPBENCH_COMMIT"; }
export PYTHONPATH="$W/MD22/src"
( cd MD22 && python -m pytest -q tests/ )

if [ -z "${CKPT_IN:-}" ] || want force-matpes; then
stage data-matpes
python MD22/scripts/prepare_data.py matpes --out data
mkdir -p hf_data && cp data/matpes_pbe_10pct_train.npz data/matpes_pbe_10pct_val.npz data/matpes_pbe_10pct_split.json hf_data/
hf_up "$HF_DATA_REPO" dataset hf_data . "MatPES-PBE v2025.1 10% subset (seed 0), 95/5 train/val"
fi

if [ -n "${CKPT_IN:-}" ]; then
  stage wait-checkpoint "$CKPT_IN"
  for _ in $(seq 1 360); do [ -s "$CKPT_IN" ] && break; sleep 30; done
  [ -s "$CKPT_IN" ] || { echo "checkpoint $CKPT_IN never arrived"; exit 1; }
  sleep 60  # let an in-progress copy finish writing
  CKPT="$CKPT_IN"
  python -c "import torch,json,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print('checkpoint history', json.dumps(c['history'][-1]))" "$CKPT"
else
  stage train
  python -m md22nop.training.train --train data/matpes_pbe_10pct_train.npz --val data/matpes_pbe_10pct_val.npz \
    --out ckpt --hours "$TRAIN_HOURS" --workers "$WORKERS" 2>&1 | tee logs/train.log
  hf_up "$HF_MODEL_REPO" model ckpt checkpoints "e2IP-NequIP checkpoints"
  CKPT="$W/ckpt/final.pt"
fi
set +e

if want force-matpes; then
stage eval-force-matpes
# Held-out MatPES-PBE structures only: the 2,174-structure validation split of the 10% subset
# (never used for gradient updates).
python MD22/scripts/eval_force.py --checkpoint "$CKPT" --dataset data/matpes_pbe_10pct_val.npz --name matpes_pbe_heldout \
  --fpbench FPBench --out results/force --workers "$WORKERS" 2>&1 | tee logs/eval_force_matpes.log
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "STAGE eval-force-matpes FAILED"
gzip -f results/force/matpes_pbe_heldout_force_results_standardized.json
hf_up "$HF_MODEL_REPO" model results results "FPBench force (MatPES-PBE)"
fi

if want phase; then
stage eval-phase
python MD22/scripts/eval_phase.py --checkpoint "$CKPT" --fpbench FPBench --out results/phase --workers "$WORKERS" 2>&1 | tee logs/eval_phase.log
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "STAGE eval-phase FAILED"
hf_up "$HF_MODEL_REPO" model results results "FPBench phase stability & ordering"
fi

if want neb; then
stage eval-neb
python MD22/scripts/eval_neb.py --checkpoint "$CKPT" --fpbench FPBench --src MD22/src --out results/neb --workers "$WORKERS" \
  --fraction "$NEB_FRACTION" --seed 0 2>&1 | tee logs/eval_neb.log
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "STAGE eval-neb FAILED"
rm -rf results/neb/neb_component/generation/runs
hf_up "$HF_MODEL_REPO" model results results "FPBench ion-migration NEB"
fi

if want omat24; then
stage data-omat24
python MD22/scripts/prepare_data.py omat24 --out data --fraction "$OMAT_FRACTION" || echo "STAGE data-omat24 FAILED"
OMAT_NPZ=data/omat24_rattled_1000.npz; OMAT_NAME=omat24_rattled_1000
if python -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1 else 1)" "$OMAT_FRACTION"; then
  PCT=$(python -c "import sys; print(int(round(float(sys.argv[1]) * 100)))" "$OMAT_FRACTION")
  OMAT_NPZ=data/omat24_rattled_1000_${PCT}pct.npz; OMAT_NAME=omat24_rattled_1000_${PCT}pct
fi

stage eval-force-omat24
python MD22/scripts/eval_force.py --checkpoint "$CKPT" --dataset "$OMAT_NPZ" --name "$OMAT_NAME" \
  --fpbench FPBench --out results/force --workers "$WORKERS" 2>&1 | tee logs/eval_force_omat24.log
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "STAGE eval-force-omat24 FAILED"
gzip -f "results/force/${OMAT_NAME}_force_results_standardized.json"
hf_up "$HF_MODEL_REPO" model results results "FPBench force (OMat24 rattled-1000)"
fi

stage done
[ -z "$SYNC_PID" ] || kill "$SYNC_PID" || true
hf_up "$HF_MODEL_REPO" model logs logs "final logs"
