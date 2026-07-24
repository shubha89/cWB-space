# Detection Statistic Checkpoint

Date: 2026-07-24

This checkpoint preserves the currently working LISA link simulation and time-frequency detection-statistic flow before further development.

## Locked Functionality

- MBHB waveform injection into LISA six-link data.
- Realistic LISA noise generation through `lisainstrument`.
- PyTDI-based `XYZ` and `AET` products.
- T-whitened WDM maps for A and E channels.
- Separate A/E tube morphology selection.
- Channel-consistency summaries:
  - `J`: A/E tube Jaccard overlap.
  - `C`: complex A/E coherence over the selected region.
  - `E_A`, `E_E`: T-whitened channel powers.
- Current detection statistic:

```text
D = J * sqrt(E_A + E_E)
```

## Current Interpretation

The present five-seed noise-only background is exploratory, not a final FAR calibration. With the current background sample, `D` separates all tested GW+noise MBHB masses except the `10^4 Msun` case. The doubled-distance cases are included in the comparison and show the expected lower-amplitude behavior, with some non-monotonicity from morphology-dependent tube selection.

## Important Local Outputs

Generated HDF5, CSV, and plot outputs are intentionally left out of git because they are large and reproducible. The key local outputs at checkpoint time are under:

```text
diagnostics/mbhb_sixlinks/outputs/
```

The current detection-statistic products are:

```text
diagnostics/mbhb_sixlinks/outputs/j_sqrt_energy_detection_statistic.csv
diagnostics/mbhb_sixlinks/outputs/plots/j_sqrt_energy_detection_statistic.png
```

## Next Development Direction

The next step should build on this checkpoint by expanding the noise-only background, estimating empirical false alarm probabilities/rates for `D`, and then iterating on morphology-aware clustering without changing this locked baseline.
