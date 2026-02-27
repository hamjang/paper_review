# SGLang Docs

> 참고: https://docs.sglang.io/advanced_features/server_arguments.html

---

## 목차

- [기본 개념](#기본-개념)
  - [1. 비트 표현 방식](#1-비트-표현-방식-precision-format)
  - [2. GPTQ](#2-gptq)
  - [3. AWQ](#3-awq)
- [Server Arguments](#server-arguments)
  - [1. Common Launch Commands](#1-common-launch-commands)
  - [2. Model and Tokenizer](#2-model-and-tokenizer)
  - [3. Quantization and Data Type](#3-quantization-and-data-type)
  - [4. Memory and Scheduling](#4-memory-and-scheduling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Quantization](#quantization)
  - [1. 양자화 개요](#1-양자화-개요)
  - [2. 오프라인 양자화](#2-오프라인-양자화)
  - [3. 모델 양자화 도구](#3-모델-양자화-도구-offline-tools)
  - [4. NVIDIA ModelOpt 활용](#4-nvidia-modelopt-활용)
  - [5. 온라인 양자화](#5-온라인-양자화)
  - [6. 알려진 제한 사항](#6-알려진-제한-사항)

---

## 기본 개념

### 1. 비트 표현 방식 (Precision Format)

- 숫자를 컴퓨터가 얼마나 정밀하게 저장하느냐를 나타내는 방식

#### FP (Floating Point, 부동소수점)

- 숫자를 **부호 + 지수 + 가수** 형태로 저장해서 매우 크거나 작은 수도 표현 가능

| 타입 | 비트 수 | 메모리 (10억 파라미터당) | 정밀도 | 용도 |
|------|---------|------------------------|--------|------|
| FP32 | 32비트 | 4GB | 높음 | 학습 기본값 |
| FP16 | 16비트 | 2GB | 중간 | 추론 일반적 |
| FP8  | 8비트  | 1GB | 낮음 | 빠른 추론 |

#### BF (Brain Float)

- Google이 개발, FP32와 지수 범위가 동일
- FP16보다 표현 범위가 넓어서 학습 안정성이 높음 → 요즘 LLM 학습/추론 표준

| 타입 | 비트 수 | 특징 |
|------|---------|------|
| BF16 | 16비트 | FP32와 지수 범위 동일, 수치 안정성 우수 |

#### 핵심 요약

| 기준 | 순서 |
|------|------|
| 정밀도 | `FP32` > `BF16` ≈ `FP16` > `FP8` > `INT8` > `INT4` |
| 메모리 | `FP32` > `BF16` = `FP16` > `FP8` = `INT8` > `INT4` |
| 속도   | `FP32` < `BF16` = `FP16` < `FP8` < `INT8` < `INT4` |

> [!TIP]
> 추론 시에는 **BF16** 또는 **FP8**을 많이 사용. FP8은 메모리를 절반으로 줄이면서 품질 손실이 크지 않아 최근 많이 사용되는 추세

---

### 2. GPTQ

> 이미 학습된 모델을 재학습 없이 저비트(INT4/INT3)로 압축하는 양자화 기법 (Post-Training Quantization)

#### 핵심 아이디어

- 일반 양자화는 단순히 숫자를 잘라내지만, GPTQ는 잘라낼 때 생기는 **오차를 주변 가중치에 분산시켜 보정**
- 수학적으로는 **Hessian 행렬**을 이용해 어떤 가중치가 중요한지 파악하고 오차를 최소화

| 항목 | 내용 |
|------|------|
| 압축률 | FP16 대비 4배 (INT4 기준) |
| 속도 | 변환 시간이 걸리지만 추론은 빠름 |
| 품질 | 단순 양자화보다 정확도 손실 적음 |
| 재학습 | 불필요 (Post-Training) |

#### 언제 쓰나?

- GPU 메모리가 부족할 때 (예: 70B 모델을 A100 1장에 올리고 싶을 때)
- Hugging Face에서 `모델명-GPTQ` 형태로 배포된 모델들이 이 방식
- vLLM/SGLang 모두 지원 (`--quantization gptq`)

---

### 3. AWQ

> "중요한 가중치는 건드리지 말자" — 활성화값을 분석해서 중요한 가중치를 보호하며 INT4로 압축 (Activation-aware Weight Quantization)

#### GPTQ와의 핵심 차이

- GPTQ는 오차를 **사후에 보정**, AWQ는 애초에 **중요한 가중치를 찾아서 보호**
- 중요도 판단 기준: **활성화값(Activation)** — 실제 데이터가 통과할 때 많이 쓰이는 가중치 = 중요한 가중치

| 항목 | 내용 |
|------|------|
| 압축률 | FP16 대비 4배 (INT4 기준) |
| 품질 | GPTQ보다 일반적으로 좋음 |
| 변환 속도 | GPTQ보다 빠름 |
| 재학습 | 불필요 |

#### GPTQ vs AWQ 비교

| 항목 | GPTQ | AWQ |
|------|------|-----|
| 접근법 | 오차 보정 | 중요 가중치 보호 |
| 품질 | 좋음 | 더 좋음 |
| 변환 시간 | 느림 | 빠름 |
| 추론 속도 | 비슷 | 비슷 |
| 메모리 | 비슷 | 비슷 |

#### 양자화 방식 전체 비교

| 방식 | 방법 | 품질 | 속도 |
|------|------|------|------|
| FP8  | 단순 비트 축소 | ★★★★ | ★★★★ |
| GPTQ | 오차 보정 INT4 | ★★★☆ | ★★★★ |
| AWQ  | 중요 가중치 보호 INT4 | ★★★★ | ★★★★ |
| GGUF | CPU 친화적 압축 | ★★★☆ | ★★☆☆ |

---

## Server Arguments

- 배포 시 언어 모델 서버의 동작과 성능을 설정하기 위해 CLI에서 사용되는 서버 인자(Arguments) 목록 제공
- 모델 선택, 병렬 처리 정책, 메모리 관리 및 최적화 기법 등 사용자 정의 가능

---

### 1. Common Launch Commands

#### 1.1 멀티 GPU 병렬 처리 (TP & DP)

- **텐서 병렬화 (Tensor Parallelism)**: 멀티 GPU 텐서 병렬 활성화 `--tp 2`
  - `"peer access is not supported..."` 에러 발생 시 `--enable-p2p-check` 추가
  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```

- **데이터 병렬화 (Data Parallelism)**: 멀티 GPU 데이터 병렬 활성화 `--dp 2`
  - TP와 함께 사용 가능하며, 메모리 충분 시 처리량 향상에 유리

---

#### 1.2 메모리 부족 (OOM) 해결 및 최적화

- **KV 캐시 메모리 조정**: 서빙 중 메모리 부족 에러 발생 시 `--mem-fraction-static` 값을 기본값(0.9)보다 작게 설정하여 KV 캐시 풀 점유율 줄임
  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
  ```

- **프롬프트 prefill 최적화**: 긴 프롬프트 처리 중 메모리가 부족하다면 `chunked prefill` 크기를 작게 설정
  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 512
  ```

- **공유 메모리 (Docker/K8s)**: 컨테이너 환경에서는 프로세스 간 통신을 위한 공유 메모리(`shm-size` 또는 `/dev/shm`) 설정 필요

---

#### 1.3 양자화 및 정밀도

- **FP8 가중치 양자화**: FP16 체크포인트에 `--quantization fp8`을 추가하거나, 이미 FP8로 양자화된 체크포인트를 직접 로드

- **FP8 KV 캐시 양자화**: `--kv-cache-dtype fp8_e4m3` 또는 `--kv-cache-dtype fp8_e5m2` 사용하여 KV 캐시 양자화 가능

---

#### 1.4 기타 고급 설정

- **결정론적 추론**: `--enable-deterministic-inference`를 추가하면 배치(batch) 구성에 상관없이 동일한 결과 보장하는 연산 수행

- **커스텀 채팅 템플릿**: 토크나이저에 템플릿이 없는 경우 커스텀 템플릿을 지정하거나, `--hf-chat-template-name`으로 특정 템플릿(예: `tool_use`)을 선택할 수 있음

- **멀티 노드 분산 실행**: 여러 대의 서버(Node)에서 실행할 때는 `--nnodes`와 `--node-rank`를 사용

---

### 2. Model and Tokenizer

#### 2.1 가중치 로드 방식 (`--load-format`)

- **`layered`**: 메모리가 부족한 환경에서 핵심. 모든 가중치를 한꺼번에 불러오지 않고 레이어별로 불러와서 양자화(Quantization)를 진행한 뒤 다음 레이어로 넘김. 이로 인해 로딩 시 발생하는 일시적인 **메모리 피크(Peak Memory)**를 크게 낮춤

- **`npcache`**: 모델을 처음 로드할 때 넘파이(NumPy) 캐시 파일 생성. 이후 서버를 재시작할 때 모델 로딩 속도가 비약적으로 빨라짐. **모델 서빙 테스트**를 자주 해야 할 때 유용함

- **`dummy`**: 실제 모델 가중치를 로드하지 않고 랜덤값으로 채움. 모델이 정상적으로 서빙되는지, 혹은 특정 사양의 GPU에서 속도가 얼마나 나오는지 **프로파일링(Profiling)**만 하고 싶을 때 사용함

---

#### 2.2 하드웨어 최적화 구현체 (`--model-impl`)

- **`sglang`** (권장): SGLang이 직접 최적화한 커널 사용. **RadixAttention** 같은 고유 기술이 적용되어 추론 속도가 빠름

- **`transformers`**: Hugging Face의 기본 구현체를 사용. SGLang에서 아직 공식 지원하지 않는 최신 모델을 돌려야 할 때 하위 호환성용으로 사용

---

#### 2.3 메모리 및 컨텍스트 관리 (`--context-length`)

- **역할**: 모델이 한 번에 처리할 수 있는 최대 토큰 수를 강제로 지정

> [!TIP]
> 모델의 기본 설정값이 128K라 하더라도, 내 서비스가 8K 정도만 사용한다면 이 값을 `8192`로 제한.
> 남은 GPU 메모리가 모두 **KV 캐시**로 할당되어, 더 많은 사용자가 동시에 접속해도 버틸 수 있는 **처리량**이 늘어남

---

#### 2.4 토크나이저 성능 최적화 (`--tokenizer-worker-num`)

- 대규모 서빙 시 병목 현상은 의외로 GPU가 아닌 CPU(토크나이저)에서 발생

- **현상**: GPU 사용률은 낮은데 응답 속도가 느리다면 토크나이징 과정이 밀리고 있을 가능성이 큼

- **해결**: `--tokenizer-worker-num`을 `4`나 `8`로 늘림. 별도의 프로세스가 토크나이징을 병렬로 처리하여 전체적인 대기 시간(Latency)을 줄여줌

---

### 3. Quantization and Data Type

#### 3.1 KV 캐시 양자화 (`--kv-cache-dtype`)

- LLM 서빙 시 GPU 메모리를 가장 많이 잡아먹는 주범은 가중치가 아니라, 문맥을 기억하기 위한 **KV 캐시**

- **설정값**: `fp8_e4m3` (NVIDIA Ada, Hopper, Ampere 아키텍처 권장)

- **왜 유용한가?**: 기본적으로 KV 캐시는 16비트(FP16/BF16)로 저장됨. 이를 8비트(FP8)로 낮추면 **캐시가 사용하는 메모리가 정확히 절반**으로 줄어듦

- **효과**: 동일한 GPU에서 **더 긴 문장(Context Length)**을 처리하거나, **더 많은 동시 접속자(Batch Size)**를 수용할 수 있음. 정확도 손실도 거의 체감되지 않는 수준임

---

#### 3.2 모델 가중치 양자화 (`--quantization`)

- 모델 자체의 덩치를 줄여서 저사양 GPU에서도 큰 모델을 돌릴 수 있게 함

- **주요 옵션**
  - `awq` / `gptq`: 4비트 양자화의 표준. 70B 모델을 A100 80GB 한 장에서 돌리고 싶을 때 필수적
  - `fp8`: 최신 가속기(H100 등)에서 성능 최적화가 가장 잘 되어 있음. 속도와 정확도의 밸런스가 가장 좋음
  - `bitsandbytes`: 별도의 양자화 과정 없이 Hugging Face 모델을 그대로 4비트/8비트로 로드할 때 편리함

> [!TIP]
> 이미 양자화된 모델(예: `Llama-3-8B-Instruct-AWQ`)을 사용할 때는 별도의 인자 없이 자동 인식되지만, 명시적으로 적어주는 것이 안전함

---

#### 3.3 데이터 타입 선택 (`--dtype`)

- 모델의 기본 연산 정밀도 결정

- **추천값**: `bfloat16` (BF16)

- **왜 유용한가?**: 최신 GPU(RTX 30/40, A100, H100 등)를 쓴다면 `float16`보다 `bfloat16`이 훨씬 유리함. `bfloat16`은 `float32`와 수치 표현 범위가 같아서, 연산 중에 값이 너무 커지거나 작아져서 발생하는 **수치적 불안정성(NaN 에러 등)을 방지**해 줌

> [!WARNING]
> 아주 오래된 GPU(V100, T4 등)는 BF16을 지원하지 않으므로 이때는 `half`나 `float16` 사용

---

#### 3.4 LM Head FP32 연산 (`--enable-fp32-lm-head`)

- **설정값**: `True` (플래그 추가)

- **왜 유용한가?**: 모델의 마지막 출력층(Logits 생성 단계)에서만 정밀도를 높게 유지하는 설정. 전체 속도에는 큰 영향을 주지 않으면서, 양자화로 인해 발생할 수 있는 텍스트 생성 오류나 품질 저하 방지

---

### 4. Memory and Scheduling

#### 4.1 GPU 메모리 점유율 조절 (`--mem-fraction-static`)

- SGLang은 실행 시 미리 GPU 메모리의 상당 부분을 예약하여 성능을 최적화

- **기능**: 전체 GPU 메모리 중 모델 가중치와 KV 캐시(메모리 풀)가 차지할 비율을 정함

- **활용**: 기본값은 **0.9(90%)**. 만약 서버 실행 직후 `Out of Memory(OOM)` 에러가 발생한다면, 이 값을 **0.8**이나 **0.7**로 낮춰서 시스템이나 다른 프로세스가 사용할 여유 메모리를 확보해야 함

> [!TIP]
> GPU를 오직 SGLang 전용으로만 쓴다면 `0.92~0.95`까지 올려서 KV 캐시 용량을 극대화할 수 있음

---

#### 4.2 긴 프롬프트 처리를 위한 조각 내기 (`--chunked-prefill-size`)

- 매우 긴 문서나 대화 기록을 입력할 때 발생하는 메모리 급증(Spike) 현상 방지

- **기능**: 프롬프트가 입력될 때 한 번에 계산하는 토큰의 최대량 제한

- **활용**: 예를 들어 32K 길이의 프롬프트가 들어올 때 이 값을 **4096**으로 설정하면, 8번(4096 × 8)에 걸쳐 나누어 처리함

- **장점**: 입력 단계(Prefill)에서 메모리가 순간적으로 폭발하여 서버가 죽는 것을 막아주며, 동시에 진행 중인 다른 사용자들의 응답(Decode)이 멈추는 현상 완화

---

#### 4.3 캐시 재사용 정책 (`--radix-eviction-policy`)

- SGLang의 전매특허인 **Radix Cache**가 가득 찼을 때, 어떤 데이터를 먼저 버릴지 결정

- **LRU (Least Recently Used)**: 가장 오랫동안 사용되지 않은 캐시를 먼저 삭제. 일반적인 챗봇이나 범용 서비스에 가장 적합

- **LFU (Least Frequently Used)**: 사용 빈도가 가장 낮은 캐시를 삭제. 특정 문서(RAG용 PDF 등)나 공통 시스템 프롬프트가 반복해서 호출되는 환경에서 성능이 좋음

- **효과**: 캐시가 적중(Hit)되면 동일한 프롬프트에 대해 계산 과정을 생략하고 즉시 응답을 시작

---

#### 4.4 동시 처리 및 대기열 제어 (`--max-running-requests` & `--schedule-policy`)

- 서버가 한 번에 얼마나 많은 일을 할지, 그리고 어떤 일을 먼저 할지 결정

- `--max-running-requests`: 동시에 연산할 최대 요청 수. 이 값을 지정하지 않으면 메모리가 허용하는 한 최대치로 작동. 너무 높으면 개별 응답 속도가 느려질 수 있음

- `--schedule-policy lpm` (Longest Prefix Match): SGLang에서 특히 유용한 정책으로, **이미 캐시된 내용과 가장 많이 겹치는 요청을 먼저 처리**함. 전체적인 GPU 연산량을 줄여 서버 효율 극대화

---

## Hyperparameter Tuning

- 오프라인 배치 추론(Offline Batch Inference)에서 **최대 처리량**을 달성하기 위한 가이드

### 1. 오프라인 배치 추론 성능 최적화

- 오프라인 배치 작업에서 처리량을 높이는 가장 중요한 요소는 **배치 크기를 최대한 크게 유지하는 것**. 서버가 정상 상태로 풀가동 중일 때, 로그에서 다음 항목 확인

  - **로그 예시**: `Decode batch. #running-req : 233, #token : 370959, token usage : 0.82, cuda graph : True, gen throughput (token/s) : 4594.01, #queue-req : 317`

---

### 2. 주요 조정 항목

#### 2.1 요청 제출 속도 조절 (`#queue-req`)

- **의미**: 대기열에 쌓인 요청 수

- **조절**: `#queue-req`가 자주 `0`이 된다면 클라이언트가 요청을 너무 느리게 보내고 있다는 뜻

- **권장 범위**: **100 ~ 2000** 유지 권장. 단, 너무 크게 설정하면 서버의 스케줄링 오버헤드 증가

---

#### 2.2 높은 토큰 사용률 달성 (`token usage`)

- **의미**: 서버의 KV 캐시 메모리 활용도. 0.9(90%) 이상이면 높은 활용도

- **너무 보수적일 때** (`token usage < 0.9` and `#queue-req > 0`): 서버가 새 요청을 받는 데 너무 조심스러운 상태. `--schedule-conservativeness`를 `0.3` 정도로 낮추기 (주로 사용자가 `max_new_tokens`를 크게 잡았지만 실제로는 일찍 끝나는 경우 발생)

- **너무 공격적일 때** (`KV cache pool is full` 경고 빈번): 캐시가 꽉 차서 요청을 취소(Retract)하는 로그가 자주 보인다면 `--schedule-conservativeness`를 `1.3` 정도로 높임

> [!NOTE]
> 분당 1회 정도의 `KV cache pool is full` 경고는 정상 범위

---

#### 2.3 KV 캐시 용량 확대를 위한 `--mem-fraction-static` 튜닝

- SGLang의 메모리 할당 구조는 다음과 같음
  **전체 메모리 = 모델 가중치 + KV 캐시 풀 + CUDA 그래프 버퍼 + Activation(활성값)**

- **목표**: 더 높은 동시 처리를 위해 Activation과 CUDA 그래프용 메모리만 남기고 **KV 캐시 풀을 최대화**해야 함

- **튜닝 방법**: 서버 시작 직전 로그에서 `available_gpu_mem` 확인
  - **5~8GB 사이**: 적절한 설정
  - **10~20GB 이상**: 메모리가 남음. `--mem-fraction-static`을 높여서 KV 캐시에 할당

> [!WARNING]
> `available_gpu_mem`이 너무 낮으면 OOM 위험 있음

---

#### 2.4 OOM 에러 방지

- **Prefill 단계 OOM**: `--chunked-prefill-size`를 **4096** 또는 **2048**로 낮춤 (속도는 조금 느려지지만 메모리 아낌)

- **Decoding 단계 OOM**: `--max-running-requests`를 줄임

- **공통**: `--mem-fraction-static`을 **0.8**이나 **0.7**로 낮추면 전체적인 메모리 사용량이 줄어들어 안전해지지만, 최대 처리량 감소

---

#### 2.5 CUDA 그래프 최적화 (`--cuda-graph-max-bs`)

- 기본적으로 CUDA 그래프는 작은 배치 사이즈(160~256)에서만 작동. 하지만 대형 모델이나 높은 TP(Tensor Parallelism) 설정에서는 배치 사이즈 **512\~768**까지도 효과적임. 이 값을 높이면 성능이 좋아지지만 메모리를 더 많이 먹으므로 `--mem-fraction-static`을 약간 낮춰야 할 수도 있음

---

#### 2.6 병렬 처리 전략 (`--dp-size` 및 `--tp-size`)

- **데이터 병렬화(DP)가 처리량에 더 유리**: 메모리가 허용된다면 항상 DP를 우선 고려

- SGLang 내장 `dp_size` 인자보다는 **SGLang Model Gateway(구 Router)**를 사용하는 것이 분산 처리에 더 효율적임

---

#### 2.7 기타 시도해 볼 옵션들

- `--enable-torch-compile`: 작은 모델, 작은 배치 사이즈에서 가속 효과가 있음

- `--quantization fp8`: FP8 양자화를 통해 메모리와 속도에 대해 동시에 이점 확보

- `--schedule-policy lpm`: 공유 프롬프트(Shared Prefix)가 많은 배치 작업인 경우, **가장 긴 접두사 일치(Longest Prefix Match)** 정책을 써서 캐시 적중률 극대화

---

## Quantization

- 양자화는 모델의 크기를 줄이고 추론 속도를 높이는 핵심 기술로, SGLang은 오프라인 및 온라인 방식의 다양한 최적화 도구 지원

### 1. 양자화 개요

- SGLang은 **오프라인 양자화**와 **온라인 동적 양자화**를 모두 지원함

- **오프라인 양자화 (Offline Quantization)**: 추론 전에 미리 양자화된 가중치 로드. GPTQ, AWQ와 같은 보정 데이터셋을 통해 통계값을 미리 계산해야 하는 방식에 필수적

- **온라인 양자화 (Online Quantization)**: 실행 중에 가중치의 최댓값/최솟값을 계산하여 실시간으로 저정밀도 형식으로 변환. 편리하지만 시작 시간이 길어지고 메모리 사용량이 일시적으로 증가

> [!WARNING]
> 더 나은 성능과 편의성을 위해 **오프라인 양자화 사용을 강력히 권장**. 이미 양자화된 모델을 쓸 때는 `--quantization` 인자 중복 추가 금지

---

### 2. 오프라인 양자화

- 이미 양자화된 모델(예: AWQ, FP8 체크포인트)을 사용할 때는 단순히 모델 경로만 지정하면 됨. 엔진이 Hugging Face 설정 파일(`config.json`)을 읽어 양자화 방식을 자동으로 파악

- **실행 예시 (AWQ 모델)**
  ```bash
  python3 -m sglang.launch_server \
      --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
      --port 30000 --host 0.0.0.0
  ```

- **고급 설정 (W8A8 FP8)**: 특정 커스텀 커널(CuTLASS 등)을 사용하고 싶다면 명시적으로 지정할 수 있음(`--quantization w8a8_fp8`)
  ```bash
  python3 -m sglang.launch_server \
      --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
      --quantization w8a8_fp8 \
      --port 30000 --host 0.0.0.0
  ```

---

### 3. 모델 양자화 도구 (Offline Tools)

- 모델을 직접 양자화하고 싶을 때 사용할 수 있는 도구

- **Unsloth**: 가장 권장되는 방식 중 하나

- **Auto-round**: LLM뿐만 아니라 VLM 양자화 지원

- **GPTQModel**: GPTQ 방식을 위한 간편한 파이썬 라이브러리

- **LLM Compressor (Neural Magic)**: FP8 동적 양자화 모델을 만들 때 유용

---

### 4. NVIDIA ModelOpt 활용

- NVIDIA 하드웨어에 최적화된 고급 양자화 기술 제공

- **이점**: VRAM 사용량 감소, 높은 처리량, 낮은 지연 시간

- **지원 형식**: `fp8` (Hopper 이상), `fp4` (Blackwell 이상)

- **ModelOpt를 이용한 양자화 및 내보내기 예시**
  ```bash
  # FP8로 양자화하여 특정 디렉토리에 저장
  python examples/usage/modelopt_quantize_and_export.py quantize \
      --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --export-dir ./quantized_tinyllama_fp8 \
      --quantization-method modelopt_fp8
  ```

---

### 5. 온라인 양자화

- 표준 BF16/FP16 모델을 로드하면서 즉석에서 양자화 옵션을 적용할 때 사용

- **FP8 온라인 양자화 실행**
  ```bash
  python3 -m sglang.launch_server \
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
      --quantization fp8 \
      --port 30000 --host 0.0.0.0
  ```

- **torchao 기반 양자화**: PyTorch의 `torchao` 라이브러리를 사용한 다양한 양자화 옵션 지원(`int8dq`, `int4wo-128` 등)
  ```bash
  python3 -m sglang.launch_server \
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
      --torchao-config int4wo-128 \
      --port 30000 --host 0.0.0.0
  ```

---

### 6. 알려진 제한 사항

- **혼합 비트 (Mixed-bit) 제한**: 동일 레이어 내에서 서로 다른 비트 정밀도를 섞어 쓰는 것은 현재 호환성 이슈가 있을 수 있음

- **MoE 모델**: 일부 MoE 모델은 커널 제한으로 인해 특정 레이어 양자화 시 오류가 발생할 수 있음

- **VLM 실패 사례**: Qwen2.5-VL 등 일부 모델에서 `auto-gptq` 형식을 쓰면 정확도가 떨어지는 현상 발생 (AWQ 형식 사용 권장)
