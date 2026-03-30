from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str], required: bool = True) -> bool:
    print(f"\n$ {' '.join(args)}")
    result = subprocess.run(args, cwd=PROJECT_ROOT, check=False)
    if result.returncode == 0:
        print("✓ passed")
        return True
    print(f"✗ failed (exit code {result.returncode})")
    if required:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run launch smoke checks for the complaint-analysis demo."
    )
    parser.add_argument(
        "--skip-app-check",
        action="store_true",
        help="Skip runtime app readiness check (token/vector store).",
    )
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Fail when app readiness check fails.",
    )
    args = parser.parse_args()

    print("Running demo smoke checks...")

    checks_ok = run_cmd([sys.executable, "-m", "src.preflight", "--mode", "ci"], required=True)
    if not checks_ok:
        return 1

    if not args.skip_app_check:
        runtime_ok = run_cmd([sys.executable, "-m", "src.preflight", "--mode", "app"], required=False)
        if not runtime_ok and args.strict_runtime:
            return 1

    tests_ok = run_cmd([sys.executable, "-m", "pytest", "-q"], required=True)
    if not tests_ok:
        return 1

    print("\nSmoke checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
