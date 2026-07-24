"""Read and query pycWB WDM cross-talk overlap catalogs."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PycWBXTalkCatalog:
    coeff: np.ndarray
    lookup_table: np.ndarray
    layers: np.ndarray
    tag: float = 0.0
    beta_order: float = 0.0
    precision: float = 0.0
    kwdm: float = 0.0

    @property
    def n_res(self) -> int:
        return int(len(self.layers))

    @classmethod
    def load(cls, path: str | Path) -> "PycWBXTalkCatalog":
        p = Path(path)
        if p.suffix == ".npz":
            data = np.load(p)
            return cls(
                coeff=np.asarray(data["xtalk_coeff"], dtype=np.float32),
                lookup_table=np.asarray(data["xtalk_lookup_table"], dtype=np.int32),
                layers=np.asarray(data["layers"], dtype=np.int32),
                tag=float(data["tag"]) if "tag" in data.files else 0.0,
                beta_order=float(data["beta_order"]) if "beta_order" in data.files else 0.0,
                precision=float(data["precision"]) if "precision" in data.files else 0.0,
                kwdm=float(data["kwdm"]) if "kwdm" in data.files else 0.0,
            )
        return cls.from_binary(p)

    @classmethod
    def from_binary(cls, path: str | Path) -> "PycWBXTalkCatalog":
        data = Path(path).read_bytes()
        offset = 0

        def unpack(fmt: str):
            nonlocal offset
            out = struct.unpack_from(fmt, data, offset)
            offset += struct.calcsize(fmt)
            return out

        n_res = int(unpack("f")[0])
        tag = beta_order = precision = kwdm = 0.0
        if n_res < 0:
            n_res = -n_res
            tag, beta_order, precision, kwdm = unpack("4f")

        layers = np.asarray([int(unpack("f")[0]) for _ in range(n_res)], dtype=np.int32)
        max_layers = int(np.max(layers))
        lookup_table = np.zeros((n_res, n_res, max_layers + 1, 2, 2), dtype=np.int32)
        coeff_rows: list[np.ndarray] = []
        entry_index = 0
        entry_dtype = np.dtype([("index", "i4"), ("coeff", "4f4")])
        entry_size = entry_dtype.itemsize

        for i in range(n_res):
            for j in range(i + 1):
                for freq1 in range(int(layers[i]) + 1):
                    for parity in range(2):
                        n_entries = int(unpack("f")[0])
                        entries = np.frombuffer(data, dtype=entry_dtype, count=n_entries, offset=offset)
                        offset += n_entries * entry_size
                        lookup_table[i, j, freq1, parity, 0] = entry_index
                        for entry in entries:
                            coeff_rows.append(
                                np.asarray(
                                    [
                                        float(entry["index"]),
                                        float(entry["coeff"][0]),
                                        float(entry["coeff"][1]),
                                        float(entry["coeff"][2]),
                                        float(entry["coeff"][3]),
                                    ],
                                    dtype=np.float32,
                                )
                            )
                            entry_index += 1
                        lookup_table[i, j, freq1, parity, 1] = entry_index

        coeff = np.vstack(coeff_rows).astype(np.float32) if coeff_rows else np.empty((0, 5), dtype=np.float32)
        return cls(
            coeff=coeff,
            lookup_table=lookup_table,
            layers=layers,
            tag=float(tag),
            beta_order=float(beta_order),
            precision=float(precision),
            kwdm=float(kwdm),
        )

    def dump_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            xtalk_coeff=self.coeff,
            xtalk_lookup_table=self.lookup_table,
            layers=self.layers,
            nRes=np.asarray(self.n_res, dtype=np.int32),
            tag=np.asarray(self.tag, dtype=float),
            beta_order=np.asarray(self.beta_order, dtype=float),
            precision=np.asarray(self.precision, dtype=float),
            kwdm=np.asarray(self.kwdm, dtype=float),
        )

    def resolution_index(self, layer: int) -> int:
        found = np.flatnonzero(self.layers == int(layer))
        if len(found) != 1:
            raise ValueError(f"Layer {layer} is not in the cross-talk catalog: {self.layers.tolist()}")
        return int(found[0])

    def get_xtalk(self, layer1: int, index1: int, layer2: int, index2: int) -> np.ndarray:
        """Return pycWB overlap coefficients for two WDM pixels.

        ``layer`` is the WDM layer count, e.g. 512 for an M=512 map.
        ``index`` is ``time_index * (layer + 1) + frequency_index``.
        """

        r1 = self.resolution_index(layer1)
        r2 = self.resolution_index(layer2)
        idx1 = int(index1)
        idx2 = int(index2)
        swap = r1 < r2
        if swap:
            r1, r2 = r2, r1
            idx1, idx2 = idx2, idx1

        layer_hi = int(self.layers[r1])
        layer_lo = int(self.layers[r2])
        time1, freq1 = divmod(idx1, layer_hi + 1)
        parity = time1 & 1
        index = idx2 - (time1 - parity) * (layer_hi // layer_lo) * (layer_lo + 1)

        start, stop = self.lookup_table[r1, r2, freq1, parity]
        rows = self.coeff[start:stop]
        if len(rows) == 0:
            return np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
        matches = rows[rows[:, 0].astype(np.int64) == int(index)]
        if len(matches) == 0:
            return np.asarray([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
        out = matches[0, 1:5].astype(np.float32)
        if swap:
            out = out.copy()
            out[1], out[2] = out[2], out[1]
        return out

    def overlap_power(self, layer1: int, index1: int, layer2: int, index2: int) -> float:
        coeff = self.get_xtalk(layer1, index1, layer2, index2)
        if coeff[0] > 2.0:
            return 0.0
        return float(np.sum(coeff * coeff))


__all__ = ["PycWBXTalkCatalog"]
