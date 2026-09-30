"""The official-test firewall, made checkable at test time rather than only asserted in prose.

Every research-extension module that touches data is required to never import or call any of the
FORBIDDEN_SYMBOLS below. This is checked by static source inspection (not just "we didn't call it in this
run"), so a future edit that reintroduces a forbidden call fails CI even if nobody happens to invoke that
code path in a given test run.
"""
from __future__ import annotations

import ast
from pathlib import Path

# Anything that can reach the official UCI test split, directly or via a script that evaluates it.
FORBIDDEN_SYMBOLS = frozenset({
    "load_dataset",        # src.data.load_dataset -- loads BOTH train and test
    "load_test_only",      # src.data.load_test_only -- loads the official test split
    "evaluate_final",       # scripts/evaluate_final.py -- the one-time final-test evaluation script
    "make_test_loader",             # src.dataset.make_test_loader
    "make_final_test_loader",       # src.dataset.make_final_test_loader
    "make_test_loader_from_stats",  # src.dataset.make_test_loader_from_stats (used only by evaluate_final.py)
})

# Modules in this package that are allowed to touch the data pipeline at all; everything else here is
# pure architecture/analysis code with no data access, so it is not scanned (nothing to check).
DATA_TOUCHING_MODULES = ("data_ext.py",)


def _names_used(source: str) -> set[str]:
    """Every identifier that appears as a Name, an attribute access, or an import in `source`."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def check_module_is_firewalled(path: Path) -> list[str]:
    """Returns the list of forbidden symbols found referenced in the file at `path` (empty = clean)."""
    source = Path(path).read_text()
    used = _names_used(source)
    return sorted(used & FORBIDDEN_SYMBOLS)


def check_package_is_firewalled(package_dir: Path) -> dict[str, list[str]]:
    """Scans every .py file directly in `package_dir` (non-recursive is enough here: this package has no
    sub-packages) and returns {relative_path: [forbidden symbols found]} for any violations."""
    violations = {}
    for py_file in sorted(Path(package_dir).glob("*.py")):
        found = check_module_is_firewalled(py_file)
        if found:
            violations[py_file.name] = found
    return violations
