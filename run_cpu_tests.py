"""Standalone CPU test runner (no pytest dependency).

Usage (from repo root, using the project venv):
    ./.venv/Scripts/python.exe run_cpu_tests.py
"""
import sys
import traceback

from tests.test_csf_squeeze import ALL_CHECKS as CORE_CHECKS
from tests.test_grid_and_rope import ALL_CHECKS as ROPE_CHECKS

ALL_CHECKS = CORE_CHECKS + ROPE_CHECKS


def main() -> int:
    passed, failed = 0, 0
    for check in ALL_CHECKS:
        name = check.__name__
        try:
            check()
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[ok]   {name}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
