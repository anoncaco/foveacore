"""Phase 0 금지 패키지 미설치/미임포트 검증기 (T1)."""

import importlib.util
import sys

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
    found = []
    for package in FORBIDDEN_PACKAGES:
        if package in sys.modules:
            found.append(package)
            continue
        try:
            available = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            # 일부 로더는 __spec__ 없이도 find_spec()에서 ValueError를 낸다.
            # 이 경우 sys.modules 확인으로 이미 적발한 경우 외에는 설치된 것으로 단정하지 않는다.
            available = False
        if available:
            found.append(package)
    if found:
        raise RuntimeError(f"금지 패키지 발견(Phase 0~7 범위 제외): {found}")
