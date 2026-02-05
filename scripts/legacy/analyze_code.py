from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


def _count_functions_classes(filepath: Path) -> dict[str, int] | None:
    try:
        content = filepath.read_text(encoding="utf-8")

        classes = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))
        functions = len(re.findall(r"^\s*def\s+\w+", content, re.MULTILINE))
        async_funcs = len(re.findall(r"^\s*async\s+def\s+\w+", content, re.MULTILINE))
        imports = len(re.findall(r"^\s*(import|from)\s+", content, re.MULTILINE))
        docstrings = content.count('"""') // 2
        type_hints = len(re.findall(r"->\s*[\w\[\],\s]+", content))
        comments = len(re.findall(r"^\s*#", content, re.MULTILINE))
        lines = len(content.splitlines())

        return {
            "classes": classes,
            "functions": functions,
            "async_functions": async_funcs,
            "imports": imports,
            "docstrings": docstrings,
            "type_hints": type_hints,
            "comments": comments,
            "lines": lines,
        }
    except Exception:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    stats: defaultdict[str, int] = defaultdict(int)
    file_count = 0

    for filepath in src_path.rglob("*.py"):
        if any(part in {"__pycache__", ".pytest_cache", ".venv", "venv"} for part in filepath.parts):
            continue

        file_stats = _count_functions_classes(filepath)
        if file_stats:
            file_count += 1
            for key, value in file_stats.items():
                stats[key] += value

    print("Code Quality Metrics for src/ directory:")
    print("=" * 60)
    print(f"Total files analyzed: {file_count}")
    print(f"Total lines of code: {stats['lines']:,}")
    print(f"Total classes: {stats['classes']}")
    print(f"Total functions: {stats['functions']}")
    print(f"Total async functions: {stats['async_functions']}")
    print(f"Total imports: {stats['imports']}")
    print(f"Total docstrings: {stats['docstrings']}")
    print(f"Total type hints: {stats['type_hints']}")
    print(f"Total comments: {stats['comments']}")
    print()
    print("Code Quality Indicators:")
    if stats["functions"] > 0:
        print(f"Documentation ratio: {stats['docstrings'] / stats['functions'] * 100:.1f}% of functions")
    if file_count > 0:
        print(f"Average lines per file: {stats['lines'] / file_count:.0f}")
        print(f"Functions per file: {stats['functions'] / file_count:.1f}")
        print(f"Classes per file: {stats['classes'] / file_count:.1f}")
        print(f"Comment density: {stats['comments'] / stats['lines'] * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
