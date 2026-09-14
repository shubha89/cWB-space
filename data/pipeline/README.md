# Observed TDI operational example

`observed_tdi.npz` contains 43,200 samples at four-second cadence: `t_lisa` in simulation-coordinate seconds, observed A2/E2/T2 and six observed Sagnac channels in Hz. This two-day sample is the exact saved continuous-noise plus native MBHB source sum for the response-study M=10^6 solar-mass, q=0.3 case. No source-only or noise-only channels are included. The manifest preserves the producing input hashes, time coordinate, original parameters and effective-inclination qualification. Parameter labels alone must not override that qualification.

From the repository root:

```sh
python examples/run_pipeline.py --output-dir outputs/pipeline --response --inverse
```

The output directory must be new. Default search settings are the response-study settings, with day `(t-t[0])/86400+4` and guarded search interval 4.25–5.75 days; these are not universal defaults for other observations. Run `--help` to select an input, key names, day origin, guard, band or JSON search configuration. Input times must be uniformly sampled and finite; optional response needs all six monitor channels in the specified order.

Search uses observed A/E only, K32 median-normalized WDM maps and retained candidate masks. Optional response uses observed monitor powers, excluding the union of all retained candidate masks in the per-frequency median baseline. Saved originals in `../analysis` are comparison fixtures; the CLI does not read them.

Optional rank-one K256 reconstruction retains the original connected-window rule (two time bins each side, 95th percentile, observed power) and raw-coefficient quadrature-average inverse. Its configurable quantile guard defaults to this sample's search guard. The transform helper adds right zero padding only if required (zero additional samples in the verified bundled run), records that count, and crops the inverse to the input length. It is a new operational follow-up example, not reproduction or validation of the paper's Figure 7. No template, injected truth, overlap target or likelihood chooses its mask. Finite-filter and record-boundary limitations remain.
