"""Generate a larger pycWB-style WDM MRA cross-talk catalog.

The bundled pycWB example catalog reaches M=1024.  This utility keeps those
coefficients unchanged and extends to larger powers of two using the WDM
scale-similarity observed in the pycWB K=M catalog.  It writes the same NPZ
schema consumed by scripts.pycwb_xtalk.PycWBXTalkCatalog.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pycwb_xtalk import PycWBXTalkCatalog

DEFAULT_SOURCE = Path("/tmp/pycwb-src/examples/injection/wdmXTalk/OverlapCatalog16-1024.bin")
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/mbhb_sixlinks/outputs/pycwb_mra_xtalk_512_4096_scale_extended.npz"


def half_frequency_bin(freq: int) -> int:
    """Map a doubled-resolution frequency bin to the nearest same-parity source bin."""

    if freq % 2 == 0:
        return freq // 2
    lower = freq // 2
    upper = lower + 1
    return lower if lower % 2 else upper


def source_bucket(catalog: PycWBXTalkCatalog, layer_hi: int, layer_lo: int, freq: int, parity: int) -> np.ndarray:
    r_hi = catalog.resolution_index(layer_hi)
    r_lo = catalog.resolution_index(layer_lo)
    start, stop = catalog.lookup_table[r_hi, r_lo, freq, parity]
    return catalog.coeff[start:stop]


def rows_to_nm(rows: np.ndarray, layer_lo: int) -> list[tuple[int, int, np.ndarray]]:
    out = []
    for row in rows:
        time_offset, freq_lo = divmod(int(row[0]), layer_lo + 1)
        out.append((time_offset, freq_lo, row[1:5].astype(np.float32)))
    return out


def scale_rows(rows: list[tuple[int, int, np.ndarray]], src_hi: int, src_lo: int, dst_hi: int, dst_lo: int, src_freq: int, dst_freq: int) -> list[tuple[int, np.ndarray]]:
    src_center = round(src_freq * src_lo / src_hi)
    dst_center = round(dst_freq * dst_lo / dst_hi)
    scaled = []
    for time_offset, src_freq_lo, coeff in rows:
        dst_freq_lo = dst_center + (src_freq_lo - src_center)
        if 0 <= dst_freq_lo <= dst_lo:
            scaled.append((time_offset * (dst_lo + 1) + dst_freq_lo, coeff))
    scaled.sort(key=lambda item: item[0])
    return scaled


def build_catalog(source: PycWBXTalkCatalog, layers: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    max_layer = max(layers)
    lookup = np.zeros((len(layers), len(layers), max_layer + 1, 2, 2), dtype=np.int32)
    coeff_rows: list[np.ndarray] = []
    entry_index = 0
    provenance_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    buckets: dict[tuple[int, int, int, int], list[tuple[int, int, np.ndarray]]] = {}

    source_layers = set(int(x) for x in source.layers.tolist())

    def ensure_pair(layer_hi: int, layer_lo: int) -> None:
        if (layer_hi, layer_lo) in provenance_by_pair:
            return
        if layer_hi in source_layers and layer_lo in source_layers:
            pair_mode = "copied_from_source"
            src_hi, src_lo = layer_hi, layer_lo
        else:
            if layer_hi % 2 or layer_lo % 2:
                raise ValueError(f"Cannot factor-two extend odd pair {layer_hi} x {layer_lo}")
            pair_mode = "scale_extended_x2"
            src_hi, src_lo = layer_hi // 2, layer_lo // 2
            ensure_pair(src_hi, src_lo)

        pair_entries = 0
        for freq_hi in range(layer_hi + 1):
            for parity in (0, 1):
                if pair_mode == "copied_from_source":
                    raw_rows = rows_to_nm(source_bucket(source, layer_hi, layer_lo, freq_hi, parity), layer_lo)
                    buckets[layer_hi, layer_lo, freq_hi, parity] = raw_rows
                else:
                    src_freq = min(half_frequency_bin(freq_hi), src_hi)
                    raw_rows = buckets[src_hi, src_lo, src_freq, parity]
                    out_rows = scale_rows(raw_rows, src_hi, src_lo, layer_hi, layer_lo, src_freq, freq_hi)
                    buckets[layer_hi, layer_lo, freq_hi, parity] = rows_to_nm(
                        np.asarray([[idx, *coeff] for idx, coeff in out_rows], dtype=np.float32),
                        layer_lo,
                    )
                pair_entries += len(buckets[layer_hi, layer_lo, freq_hi, parity])

        provenance_by_pair[layer_hi, layer_lo] = {
            "layer_hi": layer_hi,
            "layer_lo": layer_lo,
            "mode": pair_mode,
            "source_pair": [src_hi, src_lo],
            "entries": pair_entries,
        }

    for r_hi, layer_hi in enumerate(layers):
        for r_lo, layer_lo in enumerate(layers[: r_hi + 1]):
            ensure_pair(layer_hi, layer_lo)
            for freq_hi in range(layer_hi + 1):
                for parity in (0, 1):
                    lookup[r_hi, r_lo, freq_hi, parity, 0] = entry_index
                    out_rows = [
                        (n * (layer_lo + 1) + m, coeff)
                        for n, m, coeff in buckets[layer_hi, layer_lo, freq_hi, parity]
                    ]
                    for index, coeff in out_rows:
                        coeff_rows.append(np.asarray([float(index), *coeff], dtype=np.float32))
                        entry_index += 1
                    lookup[r_hi, r_lo, freq_hi, parity, 1] = entry_index

    coeff = np.vstack(coeff_rows).astype(np.float32) if coeff_rows else np.empty((0, 5), dtype=np.float32)
    provenance = [provenance_by_pair[key] for key in sorted(provenance_by_pair)]
    return coeff, lookup, provenance


def main() -> None:
    parser = ArgumentParser(description="Generate a pycWB-style MRA cross-talk catalog extended beyond M=1024.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--layers", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    args = parser.parse_args()

    source = PycWBXTalkCatalog.load(args.source)
    layers = sorted(int(x) for x in args.layers)
    coeff, lookup, provenance = build_catalog(source, layers)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        xtalk_coeff=coeff,
        xtalk_lookup_table=lookup,
        layers=np.asarray(layers, dtype=np.int32),
        nRes=np.asarray(len(layers), dtype=np.int32),
        tag=np.asarray(1.0, dtype=float),
        beta_order=np.asarray(4.0, dtype=float),
        precision=np.asarray(10.0, dtype=float),
        kwdm=np.asarray(1.0, dtype=float),
        generation_mode=np.asarray("copied_to_1024_scale_extended_above_1024"),
    )

    report = {
        "source": str(args.source),
        "output": str(output),
        "layers": layers,
        "coeff_shape": list(coeff.shape),
        "lookup_shape": list(lookup.shape),
        "generation_mode": "copied source coefficients through M=1024; scale-extended by factor two above M=1024",
        "provenance": provenance,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
