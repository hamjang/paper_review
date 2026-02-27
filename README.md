# 📄 Paper Review

LLM 서빙, 추론 최적화 관련 논문 및 프레임워크 문서 정리 저장소입니다.

---

## 목차

| 주제 | 설명 | 문서 |
|------|------|------|
| **SGLang** | LLM 서빙 프레임워크 — Server Arguments, Quantization, Hyperparameter Tuning | [바로가기](SGLang/docs.md) |

---

## SGLang

> Structured Generation Language for Large Language Models

SGLang은 LLM 추론을 위한 고성능 서빙 프레임워크입니다.
RadixAttention, Compressed FSM 등 고유 최적화 기술을 통해 빠른 추론과 효율적인 메모리 사용을 지원합니다.

### 다루는 내용

- **기본 개념** — 비트 표현 방식(FP/BF), GPTQ, AWQ
- **Server Arguments** — GPU 병렬화, 메모리 최적화, 양자화, 고급 설정
- **Hyperparameter Tuning** — 오프라인 배치 추론 최대 처리량 달성 가이드
- **Quantization** — 오프라인/온라인 양자화, NVIDIA ModelOpt, torchao

📎 [SGLang 공식 문서](https://docs.sglang.io)
