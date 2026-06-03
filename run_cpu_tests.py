"""Standalone CPU test runner (no pytest dependency, no package-name reliance).

Loads the test modules by file path to avoid clashing with any unrelated
``tests`` package that may sit on sys.path in a preconfigured cloud environment.

Usage (from the repo root):
    python run_cpu_tests.py
"""
import importlib.util
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
# Ensure the repo root is first on sys.path so `import csf_squeeze` resolves to
# the local package regardless of the environment.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load(module_name, rel_path):
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"test file not found: {path} (run from the repo root)")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    core = _load("csf_test_core", os.path.join("tests", "test_csf_squeeze.py"))
    rope = _load("csf_test_rope", os.path.join("tests", "test_grid_and_rope.py"))
    all_checks = list(core.ALL_CHECKS) + list(rope.ALL_CHECKS)

    passed, failed = 0, 0
    for check in all_checks:
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
