#!/bin/bash
#SBATCH --job-name=llm_data_gen
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/jliu043/relational_state/logs/llm_data_gen_%j.out
#SBATCH --error=/hpc2hdd/home/jliu043/relational_state/logs/llm_data_gen_%j.err
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Unified data-generation pipeline:
#   (1) build structured records for the requested SPLIT
#       -> data/structured/<split>.json
#   (2) ask the teacher LLM to fill scenario_text for that split
#       (in-place JSON update)
#
# SPLITS env variable controls which splits to build in this invocation.
# Supported: train eval_A eval_B placebo_test ood_social ood_career
# Use "all" to build every split in sequence.
# ---------------------------------------------------------------------------

set -euo pipefail

WORKDIR=/hpc2hdd/home/jliu043/relational_state
LOGDIR="${WORKDIR}/logs"
mkdir -p "${LOGDIR}"
cd "${WORKDIR}"

CONDA_ENV="${CONDA_ENV:-cnn_env}"
if [[ -f ~/.bashrc ]]; then
  # shellcheck source=/dev/null
  source ~/.bashrc
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

SPLITS="${SPLITS:-train eval_A eval_B placebo_test ood_social ood_career}"
SEED="${SEED:-20260421}"
OUTPUT_DIR="${OUTPUT_DIR:-data/structured}"

# Per-split volume knobs (CLI flags in build_structured_dataset.py).
# Defaults below match the v3 target counts; only effective when
# SKIP_STRUCTURED=0.
TRAIN_SAMPLES_PER_SCENE="${TRAIN_SAMPLES_PER_SCENE:-105}"
TRAIN_PLACEBO_SAMPLES_PER_SCENE="${TRAIN_PLACEBO_SAMPLES_PER_SCENE:-160}"
EVAL_A_SAMPLES_PER_CELL_PER_SCENE="${EVAL_A_SAMPLES_PER_CELL_PER_SCENE:-4}"
EVAL_B_PAIRS_PER_SCENE="${EVAL_B_PAIRS_PER_SCENE:-20}"
PLACEBO_TEST_SAMPLES_PER_SCENE="${PLACEBO_TEST_SAMPLES_PER_SCENE:-84}"
OOD_SOCIAL_SAMPLES_PER_SCENE="${OOD_SOCIAL_SAMPLES_PER_SCENE:-72}"
OOD_CAREER_SAMPLES_PER_SCENE="${OOD_CAREER_SAMPLES_PER_SCENE:-72}"

# Teacher-LLM knobs (write_scenarios_with_llm.py).
API_KEY="${API_KEY:-${DEEPSEEK_API_KEY:-${CHATANYWHERE_API_KEY:-}}}"
API_BASE="${API_BASE:-https://gpt-api.hkust-gz.edu.cn/v1}"
MODEL="${MODEL:-DeepSeek-R1-671B}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
MAX_WORKERS="${MAX_WORKERS:-16}"
SAVE_EVERY="${SAVE_EVERY:-50}"
JSON_INDENT="${JSON_INDENT:-2}"
LIMIT="${LIMIT:-0}"
OVERWRITE_TEACHER="${OVERWRITE_TEACHER:-0}"

SKIP_STRUCTURED="${SKIP_STRUCTURED:-0}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"

if [[ "${SKIP_TEACHER}" != "1" && -z "${API_KEY}" ]]; then
  echo "Missing API_KEY (or DEEPSEEK_API_KEY / CHATANYWHERE_API_KEY)." >&2
  echo "Set SKIP_TEACHER=1 to build structured records only." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== Unified structured + teacher pipeline ==="
echo "Start: $(date)"
echo "SPLITS=${SPLITS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SEED=${SEED}"

for SPLIT in ${SPLITS}; do
  echo ""
  echo "---- split=${SPLIT} ----"
  STRUCTURED_OUTPUT="${OUTPUT_DIR}/${SPLIT}.json"

  if [[ "${SKIP_STRUCTURED}" != "1" ]]; then
    python -u -m data.generation.build_structured_dataset \
      --split "${SPLIT}" \
      --output-dir "${OUTPUT_DIR}" \
      --seed "${SEED}" \
      --train-samples-per-scene "${TRAIN_SAMPLES_PER_SCENE}" \
      --train-placebo-samples-per-scene "${TRAIN_PLACEBO_SAMPLES_PER_SCENE}" \
      --eval-a-samples-per-cell-per-scene "${EVAL_A_SAMPLES_PER_CELL_PER_SCENE}" \
      --eval-b-pairs-per-scene "${EVAL_B_PAIRS_PER_SCENE}" \
      --placebo-test-samples-per-scene "${PLACEBO_TEST_SAMPLES_PER_SCENE}" \
      --ood-social-samples-per-scene "${OOD_SOCIAL_SAMPLES_PER_SCENE}" \
      --ood-career-samples-per-scene "${OOD_CAREER_SAMPLES_PER_SCENE}"
  fi

  if [[ "${SKIP_TEACHER}" != "1" ]]; then
    TEACHER_OVERWRITE_ARGS=()
    if [[ "${OVERWRITE_TEACHER}" == "1" ]]; then
      TEACHER_OVERWRITE_ARGS+=(--overwrite)
    fi
    python -u -m data.generation.write_scenarios_with_llm \
      --input-file "${STRUCTURED_OUTPUT}" \
      --output-file "${STRUCTURED_OUTPUT}" \
      --api-key "${API_KEY}" \
      --api-base "${API_BASE}" \
      --model "${MODEL}" \
      --temperature "${TEMPERATURE}" \
      --max-retries "${MAX_RETRIES}" \
      --request-timeout "${REQUEST_TIMEOUT}" \
      --max-workers "${MAX_WORKERS}" \
      --save-every "${SAVE_EVERY}" \
      --json-indent "${JSON_INDENT}" \
      --limit "${LIMIT}" \
      "${TEACHER_OVERWRITE_ARGS[@]}"
  fi
done

echo ""
echo "End: $(date)"
echo "Done."
