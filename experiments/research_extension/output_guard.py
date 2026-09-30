"""Output-path guard. Every directory the extension writes to must resolve to a location INSIDE
`results/research_extension/`, and must never be (or lie inside) a frozen root. Also refuses to start if
any run artifact it would create already exists (unless explicitly told to overwrite EXTENSION outputs).
"""
from __future__ import annotations

from pathlib import Path

EXTENSION_ROOT_REL = Path("results") / "research_extension"
FROZEN_ROOTS_REL = (Path("results") / "checkpoints", Path("results") / "metrics", Path("results") / "figures",
                    Path("results") / "smoke_test", Path("results") / "experiment_log.csv",
                    Path("results") / "phase1_console.txt", Path("configs"),
                    Path("report") / "report.md", Path("report") / "step6_streaming_analysis.md")


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_output_dir_safe(out_dir, project_root) -> Path:
    """Raise unless `out_dir` is inside <project_root>/results/research_extension and not inside/equal to
    any frozen root. Returns the resolved path."""
    root = Path(project_root)
    out = Path(out_dir)
    out = out if out.is_absolute() else root / out
    ext_root = root / EXTENSION_ROOT_REL
    if not _inside(out, ext_root):
        raise PermissionError(f"Refusing to write outside {ext_root}: {out}")
    for frozen in FROZEN_ROOTS_REL:
        f = root / frozen
        if _inside(out, f) or _inside(f, out):
            raise PermissionError(f"Refusing: {out} overlaps frozen path {f}")
    return out.resolve()


def assert_no_existing_outputs(paths, overwrite: bool) -> None:
    """Fail loudly (rather than silently overwrite) if any planned extension output already exists."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError("Extension outputs already exist (pass --overwrite_extension_outputs to replace "
                              "THEM; frozen files are never overwritable):\n  " + "\n  ".join(existing[:10]))
