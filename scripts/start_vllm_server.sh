#!/bin/bash
# Start vLLM server for SEF experiments
#
# Paper models (Section 4.1, Appendix A.5):
#   deepseek-14b  - DeepSeek-R1-Distill-Qwen-14B
#   gemma-12b     - Gemma 3 12B IT
#   ministral-14b - Ministral-3-14B-Instruct-2512
#   qwen-14b      - Qwen 2.5 14B Instruct
#
# Usage: ./start_vllm_server.sh [model_name] [port] [tp_size] [model_dir]
#
# Examples:
#   ./start_vllm_server.sh deepseek-14b
#   ./start_vllm_server.sh gemma-12b 8001

# Determine Python interpreter
if [ -n "$PYTHON" ]; then
    : # Use provided PYTHON
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON=$(which python3 || which python)
fi

MODEL_NAME="${1:-deepseek-14b}"
PORT="${2:-8000}"
MODEL_DIR="${4:-${HF_HOME:-$HOME/.cache/huggingface}}"

case $MODEL_NAME in
    deepseek-14b|deepseek_r1_14b)
        HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        LOCAL_DIR="DeepSeek-R1-Distill-Qwen-14B"
        DEFAULT_TP=1
        MAX_LEN=8192
        GPU_MEM=0.70
        ;;
    gemma-12b|gemma_3_12b)
        HF_MODEL="google/gemma-3-12b-it"
        LOCAL_DIR="gemma-3-12b-it"
        DEFAULT_TP=1
        MAX_LEN=8192
        GPU_MEM=0.65
        ;;
    ministral-14b|ministral_14b)
        HF_MODEL="mistralai/Ministral-3-14B-Instruct-2512"
        LOCAL_DIR="Ministral-3-14B-Instruct-2512"
        DEFAULT_TP=1
        MAX_LEN=8192
        GPU_MEM=0.70
        ;;
    qwen-14b|qwen_2_5_14b)
        HF_MODEL="Qwen/Qwen2.5-14B-Instruct"
        LOCAL_DIR="Qwen2.5-14B-Instruct"
        DEFAULT_TP=1
        MAX_LEN=8192
        GPU_MEM=0.70
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo ""
        echo "Available models (paper Section 4.1):"
        echo "  deepseek-14b   - DeepSeek-R1-Distill-Qwen-14B"
        echo "  gemma-12b      - Gemma 3 12B IT"
        echo "  ministral-14b  - Ministral-3-14B-Instruct-2512"
        echo "  qwen-14b       - Qwen 2.5 14B Instruct"
        exit 1
        ;;
esac

TP_SIZE="${3:-$DEFAULT_TP}"
GPU_MEM="${VLLM_GPU_MEM:-$GPU_MEM}"

LOCAL_PATH="${MODEL_DIR}/${LOCAL_DIR}"
if [ -d "$LOCAL_PATH" ]; then
    MODEL_TO_USE="$LOCAL_PATH"
    echo "Using local model: $LOCAL_PATH"
else
    MODEL_TO_USE="$HF_MODEL"
    echo "Using HuggingFace model: $HF_MODEL"
fi

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model: $MODEL_TO_USE"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Max Model Length: $MAX_LEN"
echo "GPU Memory Utilization: $GPU_MEM"
echo "=========================================="

export HF_HOME="${MODEL_DIR}"
export TRANSFORMERS_CACHE="${MODEL_DIR}"

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_TO_USE" \
    --tensor-parallel-size $TP_SIZE \
    --dtype bfloat16 \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_MEM \
    --port $PORT \
    --trust-remote-code
