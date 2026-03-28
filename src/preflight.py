from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"
PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "complaint_embeddings.parquet"


def validate_required_files() -> List[str]:
    required_files = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "runtime.txt",
        PROJECT_ROOT / "src" / "runtime.py",
    ]
    missing = [str(file_path) for file_path in required_files if not file_path.exists()]
    return [f"Missing required file: {path}" for path in missing]


def validate_env(required_env: List[str] | None = None) -> List[str]:
    required_env = required_env or ["HUGGINGFACEHUB_API_TOKEN"]
    errors: List[str] = []
    for env_name in required_env:
        if not os.getenv(env_name):
            errors.append(f"Missing environment variable: {env_name}")
    return errors


def validate_vector_store(vector_store_dir: Path | None = None) -> List[str]:
    vector_store_dir = vector_store_dir or VECTOR_STORE_DIR
    if not vector_store_dir.exists():
        return [f"Vector store directory not found: {vector_store_dir}"]
    if not any(vector_store_dir.iterdir()):
        return [f"Vector store directory is empty: {vector_store_dir}"]
    return []


def validate_parquet(parquet_path: Path | None = None) -> List[str]:
    parquet_path = parquet_path or PARQUET_PATH
    if not parquet_path.exists():
        return [f"Parquet file not found: {parquet_path}"]
    return []


def run_preflight(mode: str) -> List[str]:
    errors: List[str] = []

    if mode == "ci":
        errors.extend(validate_required_files())
        return errors

    if mode == "app":
        errors.extend(validate_required_files())
        errors.extend(validate_env())
        errors.extend(validate_vector_store())
        return errors

    if mode == "build":
        errors.extend(validate_required_files())
        errors.extend(validate_parquet())
        return errors

    return [f"Unsupported preflight mode: {mode}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Project preflight checks")
    parser.add_argument(
        "--mode",
        choices=["ci", "app", "build"],
        default="ci",
        help="ci=repo checks, app=runtime checks, build=embedding-build checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit JSON output",
    )
    args = parser.parse_args()

    errors = run_preflight(args.mode)
    status = "ok" if not errors else "error"

    if args.json_output:
        print(json.dumps({"status": status, "mode": args.mode, "errors": errors}, indent=2))
    else:
        print(f"Preflight mode: {args.mode}")
        if errors:
            print("Status: error")
            for error in errors:
                print(f"- {error}")
        else:
            print("Status: ok")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
