"""forbidden_packages.py 검증 테스트 (T1)."""

import sys
from types import ModuleType

import pytest

import foveacore.forbidden_packages as forbidden
from foveacore.forbidden_packages import FORBIDDEN_PACKAGES, assert_no_forbidden_packages


def test_no_forbidden_packages_installed():
    assert_no_forbidden_packages()


def test_forbidden_list_covers_contract():
    assert {
        "transformers", "openai", "anthropic", "llama_cpp", "vllm", "langchain",
    }.issubset(FORBIDDEN_PACKAGES)


def test_importable_forbidden_package_is_rejected(monkeypatch):
    for package in FORBIDDEN_PACKAGES:
        monkeypatch.delitem(sys.modules, package, raising=False)

    def find_spec(name):
        return object() if name == "openai" else None

    monkeypatch.setattr(forbidden.importlib.util, "find_spec", find_spec)
    with pytest.raises(RuntimeError, match="openai"):
        assert_no_forbidden_packages()


def test_loaded_module_without_spec_is_rejected(monkeypatch):
    module = ModuleType("openai")
    module.__spec__ = None
    monkeypatch.setitem(sys.modules, "openai", module)
    with pytest.raises(RuntimeError, match="openai"):
        assert_no_forbidden_packages()


def test_find_spec_value_error_does_not_mask_other_packages(monkeypatch):
    for package in FORBIDDEN_PACKAGES:
        monkeypatch.delitem(sys.modules, package, raising=False)

    def find_spec(name):
        if name == "openai":
            raise ValueError("__spec__ is None")
        return object() if name == "langchain" else None

    monkeypatch.setattr(forbidden.importlib.util, "find_spec", find_spec)
    with pytest.raises(RuntimeError, match="langchain"):
        assert_no_forbidden_packages()
