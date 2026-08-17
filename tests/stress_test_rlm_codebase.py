"""
OTF-LLM Engine: Codebase Architecture RLM Stress-Test
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import time

# Add repository root to python module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from otf_llm.run_rlm_file import run_rlm_on_file


def dump_entire_codebase(output_file: str = "data/full_codebase.txt") -> str:
    """
    Recursively scans the project repository and aggregates all source files into a single document.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    valid_extensions = (".py", ".md", ".toml", ".json", ".txt")
    ignored_dirs = {".git", ".venv", "models", "__pycache__", "build", "dist", "otf_llm.egg-info"}

    collected_content = []
    total_files = 0

    print("\n📦 [1/3] Scanning and packaging entire repository codebase...", flush=True)

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            if f.endswith(valid_extensions) and not f.startswith("."):
                file_path = os.path.join(root, f)
                rel_path = os.path.relpath(file_path, repo_root)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as src:
                        code = src.read()

                    collected_content.append(f"\n{'=' * 70}\nFILE: {rel_path}\n{'=' * 70}\n{code}\n")
                    total_files += 1
                except Exception as e:
                    print(f"⚠️ Warning reading {rel_path}: {e}")

    full_text = "\n".join(collected_content)
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(full_text)

    file_size_kb = len(full_text) / 1024
    total_lines = len(full_text.splitlines())

    print(f"✅ Packaged {total_files} source files into '{output_file}'")
    print(f"📊 Total Size: {file_size_kb:.1f} KB | Total Lines: {total_lines:,} lines")
    return output_file


def run_codebase_stress_test():
    print("=" * 75)
    print("🚀 GT Labs AI — Real Codebase Architecture RLM Stress Test")
    print("=" * 75)

    # 1. Aggregate repository files
    dump_path = dump_entire_codebase()

    # 2. Define complex multi-part architectural query
    query = (
        "Search the codebase to find: "
        "1) The exact 4 centroid values for LLOYD_MAX_2BIT_CENTROIDS, "
        "2) The author's official contact email, "
        "3) What is Taboo Rule #4 regarding to_empty()?"
    )

    # 3. Launch RLM Engine on the entire codebase
    print(f"\n🧠 [2/3] Launching OTF-RLM Engine on {dump_path}...")
    start_time = time.time()

    run_rlm_on_file(
        model_dir="./models/Qwen-3B-2Bit",
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        file_path=dump_path,
        query=query,
        max_turns=3
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total Codebase Audit Time: {elapsed:.2f} seconds")
    print("=" * 75)


if __name__ == "__main__":
    run_codebase_stress_test()