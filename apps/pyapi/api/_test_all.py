"""Run all Python engine tests. Run: py apps/web/api/_test_all.py"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
failed = 0
for name in ("_test_engine.py", "_test_conclude.py", "_test_backend.py"):
    print(f"\n=== {name} ===")
    rc = subprocess.run([sys.executable, os.path.join(here, name)], cwd=here).returncode
    if rc != 0:
        failed += 1

print("\n" + ("SOME SUITES FAILED" if failed else "ALL SUITES PASSED"))
sys.exit(1 if failed else 0)
