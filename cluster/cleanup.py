"""Cleanup utilities for large per-source simulation intermediates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HEAVY_RELATIVE_FILES: tuple[str, ...] = (
    "inputs/orbits.h5",
    "inputs/waveform_aligned.npz",
    "outputs/gws.h5",
    "outputs/lisa_instrument_output.h5",
    "outputs/six_links_gw_only.csv",
    "outputs/six_links_noisy.csv",
    "outputs/tdi_gw_only_XYZ_AET.csv",
    "outputs/tdi_michelson_XYZ_AET.csv",
)


@dataclass(frozen=True)
class CleanupResult:
    work_dir: Path
    deleted_files: tuple[Path, ...]
    deleted_bytes: int
    skipped: bool
    reason: str = ""


def discover_completed_work_dirs(runs_root: str | Path) -> list[Path]:
    """Return source work dirs that have completed simulation diagnostics."""

    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "outputs" / "cluster_simulation_diagnostics.json").exists()
    )


def cleanup_work_dir(
    work_dir: str | Path,
    *,
    apply: bool = False,
    require_diagnostics: bool = True,
    extra_relative_files: Iterable[str] = (),
) -> CleanupResult:
    """Remove large files from one completed source directory.

    By default this is a dry run. Set ``apply=True`` to delete files.
    """

    work = Path(work_dir)
    diagnostics = work / "outputs" / "cluster_simulation_diagnostics.json"
    if require_diagnostics and not diagnostics.exists():
        return CleanupResult(work, (), 0, True, "missing cluster_simulation_diagnostics.json")

    relative_files = (*HEAVY_RELATIVE_FILES, *tuple(extra_relative_files))
    candidates = [work / rel for rel in relative_files]
    existing = tuple(path for path in candidates if path.exists() and path.is_file())
    deleted_bytes = sum(path.stat().st_size for path in existing)
    if apply:
        for path in existing:
            path.unlink()
    return CleanupResult(work, existing, deleted_bytes, False)


def cleanup_work_dirs(
    work_dirs: Iterable[str | Path],
    *,
    apply: bool = False,
    require_diagnostics: bool = True,
) -> list[CleanupResult]:
    return [
        cleanup_work_dir(work_dir, apply=apply, require_diagnostics=require_diagnostics)
        for work_dir in work_dirs
    ]


def summarize_cleanup(results: Iterable[CleanupResult]) -> dict[str, int]:
    results = list(results)
    return {
        "work_dirs": len(results),
        "skipped": sum(1 for result in results if result.skipped),
        "files": sum(len(result.deleted_files) for result in results),
        "bytes": sum(result.deleted_bytes for result in results),
    }


def format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"
