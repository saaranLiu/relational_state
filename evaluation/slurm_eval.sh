#!/bin/bash
#SBATCH --job-name=relstate_eval
#SBATCH --partition=a128m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/jliu043/relational_state/logs/relstate_eval_%j.out
#SBATCH --error=/hpc2hdd/home/jliu043/relational_state/logs/relstate_eval_%j.err
#SBATCH --export=ALL

# ---------------------------------------------------------------------------
# Run the unified relational-state eval runner against an OpenAI-compatible
# chat endpoint.
#
# SPLITS env variable selects which test splits to evaluate in this run.
# Supported: eval_A eval_B placebo_test ood_social ood_career
# Default: all of them.
#
# MODEL_PRESET picks the model name only for HKUST-GZ presets; ds671b / Qwen /
# gpt-4-turbo / gpt-4 share one OpenAI-compatible gateway by default.
#
#   HKUST_OPENAI_API_BASE  default base URL (default: gpt-api.hkust-gz)
#   API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, AIGC_API_KEY
#       any one may hold the same campus key (first non-empty wins)
#   API_BASE               overrides the default base for all cloud presets
#   MODEL                  overrides the default model id per preset
#   MAX_RETRIES, REQUEST_TIMEOUT  passed through to eval_runner
#   RELSTATE_EVAL_TEMPERATURE  LLM sampling temperature (default 0.8).
#       Do not rely on generic TEMPERATURE: many login envs export TEMPERATURE=0.2,
#       which would override any script default if we used ${TEMPERATURE:-...}.
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p /hpc2hdd/home/jliu043/relational_state/logs

_EVAL_ENV_SAVE() {
  if [[ "${SPLITS+x}" = x ]]; then _E_SPLITS="$SPLITS"; _E_HAVE_SPLITS=1; fi
  if [[ "${MODEL_PRESET+x}" = x ]]; then _E_MODEL_PRESET="$MODEL_PRESET"; _E_HAVE_MODEL_PRESET=1; fi
  if [[ "${MAX_WORKERS+x}" = x ]]; then _E_MAX_WORKERS="$MAX_WORKERS"; _E_HAVE_MAX_WORKERS=1; fi
  if [[ "${STRUCTURED_DIR+x}" = x ]]; then _E_STRUCTURED_DIR="$STRUCTURED_DIR"; _E_HAVE_STRUCTURED_DIR=1; fi
}
_EVAL_ENV_RESTORE() {
  if [[ "${_E_HAVE_SPLITS:-0}" -eq 1 ]]; then SPLITS="${_E_SPLITS}"; fi
  if [[ "${_E_HAVE_MODEL_PRESET:-0}" -eq 1 ]]; then MODEL_PRESET="${_E_MODEL_PRESET}"; fi
  if [[ "${_E_HAVE_MAX_WORKERS:-0}" -eq 1 ]]; then MAX_WORKERS="${_E_MAX_WORKERS}"; fi
  if [[ "${_E_HAVE_STRUCTURED_DIR:-0}" -eq 1 ]]; then STRUCTURED_DIR="${_E_STRUCTURED_DIR}"; fi
}

_EVAL_ENV_SAVE
if [[ -f ~/.bashrc ]]; then source ~/.bashrc; fi
_EVAL_ENV_RESTORE

CONDA_ENV="${CONDA_ENV:-cnn_env}"
# Non-interactive batch jobs do not load `conda init`; hook then activate.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
RELSTATE_EVAL_TEMPERATURE="${RELSTATE_EVAL_TEMPERATURE:-0.8}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

cd /hpc2hdd/home/jliu043/relational_state

echo "=== Relational-state eval ==="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"

MODEL_PRESET="${MODEL_PRESET:-ds671b}"
MAX_WORKERS="${MAX_WORKERS:-16}"
STRUCTURED_DIR="${STRUCTURED_DIR:-data/structured}"
SPLITS="${SPLITS:-eval_A eval_B placebo_test ood_social ood_career}"

# One gateway + one key for all campus OpenAI-compatible presets (override with API_BASE / API_KEY).
HKUST_OPENAI_API_BASE="${HKUST_OPENAI_API_BASE:-https://gpt-api.hkust-gz.edu.cn/v1}"

case "${MODEL_PRESET}" in
  ds671b|deepseek_671)
    MODEL="${MODEL:-DeepSeek-R1-671B}"
    ;;
  hkust_qwen|qwen)
    MODEL="${MODEL:-Qwen}"
    ;;
  hkust_gpt4_turbo|gpt4_turbo)
    MODEL="${MODEL:-gpt-4-turbo}"
    ;;
  hkust_gpt4|gpt4|gpt-4)
    MODEL="${MODEL:-gpt-4}"
    ;;
  local_vllm)
    MODEL="${MODEL:-qwen-sft}"
    API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
    API_KEY="${API_KEY:-EMPTY}"
    ;;
  *)
    echo "Unsupported MODEL_PRESET: ${MODEL_PRESET}" >&2
    echo "Use one of: ds671b, qwen, gpt4_turbo, gpt4, local_vllm" >&2
    exit 1
    ;;
esac

if [[ "${MODEL_PRESET}" != "local_vllm" ]]; then
  API_BASE="${API_BASE:-${HKUST_OPENAI_API_BASE}}"
  API_KEY="${API_KEY:-${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-${AIGC_API_KEY:-}}}}"
fi

if [[ -z "${MODEL}" || -z "${API_BASE}" || -z "${API_KEY}" ]]; then
  echo "MODEL / API_BASE / API_KEY must be set for MODEL_PRESET=${MODEL_PRESET}" >&2
  exit 1
fi

MODEL_SUBDIR="${MODEL_SUBDIR:-}"
if [[ -z "${MODEL_SUBDIR}" ]]; then
  case "${MODEL_PRESET}" in
    ds671b|deepseek_671)      MODEL_SUBDIR="DeepSeek-R1-671B" ;;
    hkust_qwen|qwen)          MODEL_SUBDIR="Qwen" ;;
    hkust_gpt4_turbo|gpt4_turbo) MODEL_SUBDIR="gpt-4-turbo" ;;
    hkust_gpt4|gpt4|gpt-4)      MODEL_SUBDIR="gpt-4" ;;
    local_vllm)               MODEL_SUBDIR="local-vllm" ;;
    *)                         MODEL_SUBDIR="model" ;;
  esac
fi

OUTPUT_ROOT="evaluation/outputs/${MODEL_SUBDIR}"
mkdir -p "${OUTPUT_ROOT}"

export EVAL_PERSONA_WORKERS="${MAX_WORKERS}"

echo "MODEL=${MODEL}"
echo "API_BASE=${API_BASE}"
echo "STRUCTURED_DIR=${STRUCTURED_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SPLITS=${SPLITS}"
echo "MAX_WORKERS=${MAX_WORKERS} MAX_RETRIES=${MAX_RETRIES} REQUEST_TIMEOUT=${REQUEST_TIMEOUT} TEMPERATURE=${RELSTATE_EVAL_TEMPERATURE} MAX_TOKENS=${MAX_TOKENS}"

for SPLIT in ${SPLITS}; do
  INPUT_FILE="${STRUCTURED_DIR}/${SPLIT}.json"
  OUTPUT_FILE="${OUTPUT_ROOT}/${SPLIT}_predictions.jsonl"
  SUMMARY_FILE="${OUTPUT_ROOT}/${SPLIT}_summary.json"

  if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Skipping ${SPLIT}: ${INPUT_FILE} not found." >&2
    continue
  fi

  echo ""
  echo "---- split=${SPLIT} ----"
  echo "input=${INPUT_FILE}"
  echo "output=${OUTPUT_FILE}"

  python -u -m evaluation.eval_runner \
    --input-file "${INPUT_FILE}" \
    --output-file "${OUTPUT_FILE}" \
    --summary-file "${SUMMARY_FILE}" \
    --api-key "${API_KEY}" \
    --api-base "${API_BASE}" \
    --model "${MODEL}" \
    --max-workers "${MAX_WORKERS}" \
    --max-retries "${MAX_RETRIES}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --temperature "${RELSTATE_EVAL_TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}"
done

echo ""
echo "End: $(date)"
