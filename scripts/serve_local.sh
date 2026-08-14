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

# 환경의 bin 을 PATH 맨 앞에 올린다. python 을 절대경로로 부르면 그 안에서
# 실행되는 보조 도구(ninja 등)를 찾지 못해 커널 컴파일 단계에서 죽는다 —
# 증상이 "FileNotFoundError: 'ninja'" 라 원인이 잘 안 보인다.
export PATH="$(dirname "$VLLM_PY"):$PATH"

# 이 장비의 시스템 nvcc 는 CUDA 11.2 인데 flashinfer 는 12 이상을 요구한다.
# 커널을 JIT 컴파일하는 단계에서 "CUDA versions below 12 are not supported" 로
# 죽으므로, 컴파일이 필요한 sampler 를 끈다. 성능 최적화 하나를 포기하는 것이지
# 기능이 빠지는 것은 아니다.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

# 그래도 무언가 컴파일해야 할 때를 대비해 pip 로 딸려온 최신 CUDA 를 가리킨다.
_BUNDLED_CUDA=$(dirname "$(dirname "$VLLM_PY")")/lib/python3.12/site-packages/nvidia/cu13
if [ -x "$_BUNDLED_CUDA/bin/nvcc" ]; then
  export CUDA_HOME="$_BUNDLED_CUDA"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

echo "모델 : $MODEL"
echo "주소 : http://127.0.0.1:$PORT/v1"
echo "GPU  : $GPU"
echo

# --enable-auto-tool-choice / --tool-call-parser 가 **핵심이다.**
# 이것 없이 띄우면 모델이 도구를 부르려 해도 API 가 그냥 텍스트로 돌려주고,
# 증상이 "모델이 도구를 안 쓴다"로 보여서 모델 탓으로 오해하기 쉽다.
# 이 앱은 도구 호출이 안 되면 상담 에이전트가 통째로 죽는다.
#
# --chat-template 도 마찬가지로 필수다. 우리가 쓰는 미러(NousResearch)의
# tokenizer_config.json 에는 **도구 지원이 없는 348자짜리 최소 템플릿**만 들어
# 있다. 그대로 띄우면 모델이 `simulate_plan(생활비=500000)` 처럼 도구 호출을
# 텍스트로 뱉고, API 는 그것을 그냥 답변으로 돌려준다.
TEMPLATE=${TEMPLATE:-$(dirname "$0")/templates/llama3.1_json_tools.jinja}
TEMPLATE_ARG=()
if [ -f "$TEMPLATE" ]; then
  TEMPLATE_ARG=(--chat-template "$TEMPLATE")
  echo "템플릿: $TEMPLATE"
else
  echo "⚠️  도구용 채팅 템플릿이 없습니다: $TEMPLATE" >&2
  echo "    도구 호출이 동작하지 않을 수 있습니다." >&2
fi

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
  "${TEMPLATE_ARG[@]}" \
  "$@"
