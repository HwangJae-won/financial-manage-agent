#!/usr/bin/env bash
# 로컬 LLM 서버 (vLLM, OpenAI 호환).
#
# vLLM 은 **별도 conda 환경**에 있다. finagent 환경에 넣으면 pydantic·fastapi·numpy
# 를 자기 버전으로 끌어와 기존 테스트가 깨진다. 앱은 이 서버와 HTTP 로만 붙으므로
# 환경을 나눠도 아무 문제가 없다.
#
# 사용:
#   bash scripts/serve_local.sh            # 기본 모델·포트로 기동
#   MODEL=... PORT=... bash scripts/serve_local.sh
#
# 앱 쪽 설정은 .env 에:
#   FINAGENT_LLM_PROVIDER=local
#   LOCAL_BASE_URL=http://127.0.0.1:8001/v1
#   LOCAL_MODEL=<아래 MODEL 과 같은 값>
set -euo pipefail

VLLM_PY=${VLLM_PY:-/scratch/hpc201a02/.conda/envs/vllm/bin/python}
MODEL=${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}
PORT=${PORT:-8001}
GPU=${GPU:-0}

if [ ! -x "$VLLM_PY" ]; then
  echo "vLLM 환경이 없습니다: $VLLM_PY" >&2
  echo "먼저 만들어 주세요: conda create -y -n vllm python=3.11 && pip install vllm" >&2
  exit 1
fi

echo "모델 : $MODEL"
echo "주소 : http://127.0.0.1:$PORT/v1"
echo "GPU  : $GPU"
echo

# --enable-auto-tool-choice / --tool-call-parser 가 **핵심이다.**
# 이것 없이 띄우면 모델이 도구를 부르려 해도 API 가 그냥 텍스트로 돌려주고,
# 증상이 "모델이 도구를 안 쓴다"로 보여서 모델 탓으로 오해하기 쉽다.
# 이 앱은 도구 호출이 안 되면 상담 에이전트가 통째로 죽는다.
export CUDA_VISIBLE_DEVICES=$GPU
exec "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  "$@"
