# Reconstruction figure data

Run `python examples/plot_reconstruction.py` from the repository root (or use an absolute script path). Requires NumPy and Matplotlib. Outputs default to `outputs/`; override with `--output-dir DIRECTORY`.

These are compact plotting products for Figures 7, 8 and 10 of arXiv:2609.12986v1. They are not inputs sufficient to rerun waveform generation, detection, mask growth or inverse transforms. Float64 vectors are stored losslessly; local waveforms have not been decimated. `plot_metadata.json` records display intervals and annotations. `metrics.json` contains previously computed checks, each with its original comparison domain. A displayed zoom is not interchangeable with a full-record metric.

- `figure7.npz`: three sources, `mbhb1`–`mbhb3`; days, source A2/E2 and raw observed-cluster reconstruction. Zoom is source combined-A/E amplitude peak ±0.008 day. Units: Hz. These are the original known-window reconstruction results, without fitted amplitude scaling.
- `figure8.npz`: `case01`–`case03`; exact source and expanded-region reconstruction samples within each fixed enlarged time window, in Hz. Separate full-record context vectors use nonoverlapping 512-sample maxima of `hypot(A2_source,E2_source)`, with bin-midpoint timestamps. This envelope is a visualization, not a replacement for the original time series.
- `figure10.npz`: `rank01`–`rank06`; days, A/E aggregate injected MBHB contribution and expanded-region reconstruction within catalog coalescence ±0.012 day. Units: dimensionless fractional frequency. The reference is aggregate MBHB-only TDI, not an individually isolated source. Vertical markers distinguish catalog coalescence and original candidate-center bounds.

`*_new` denotes the fixed expanded-region reconstruction; it is not a fitted or shifted waveform. Metrics preserve both original and expanded masks and their explicitly named domains, including unfavorable outcomes. No population efficiency or false-alarm calibration is inferred.
