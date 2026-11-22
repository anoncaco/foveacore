"""Phase 0 금지 패키지 미설치/미임포트 검증기 (T1)."""

import importlib.util

FORBIDDEN_PACKAGES = [
    "transformers",
    "openai",
    "anthropic",
    "llama_cpp",
    "vllm",
    "langchain",
]


def assert_no_forbidden_packages():
    """금지 패키지가 설치/임포트 가능한지 없는지 검증(RuntimeError).

    텍스트 모델·LLM 라이브러리는 Phase 0~7 기본 프로젝트의 범위에서 배제된다(v1.9.3).
    """
    found = [
        p
        for p in FORBIDDEN_PACKAGES
        if importlib.util.find_spec(p) is not None
    ]
    if found:
        raise RuntimeError(f"금지 패키지 발견(Phase 0~7 범위 제외): {found}")
