"""forbidden_packages.py 검증 테스트 (T1)."""

from foveacore.forbidden_packages import FORBIDDEN_PACKAGES, assert_no_forbidden_packages


def test_no_forbidden_packages_installed():
    assert_no_forbidden_packages()


def test_forbidden_list_not_empty():
    assert len(FORBIDDEN_PACKAGES) >= 1
