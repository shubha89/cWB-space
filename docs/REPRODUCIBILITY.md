# Reproducibility scope

The reference is **arXiv:2609.12986v1**, submitted 11 September 2026. All ten local figure PDFs were compared with the public arXiv source archive on 14 September 2026 and are byte-identical. Their hashes are recorded in `data/published_figures.json`.

## Three distinct operations

1. **Replotting**: the examples regenerate nine numerical figures from their stored numerical inputs. Displayed waveform samples are retained at their original precision and cadence. The precessing full-record context uses the same 512-sample/2048-second maximum envelope as the paper. Figure 1 is copied from the original artwork; its editable source has not been located.
2. **Saved-map algorithm replay**: the included analysis example repeats event selection and ranking from normalized A/E WDM maps, then recomputes observed-background response sums and their ratio. Tests compare masks, all stored event scalars and response sums against original results. A separate saved-power test repeats the expanded reconstruction selection.
3. **Observed-TDI pipeline and original study workflows**: `examples/run_pipeline.py` performs actual WDM search, response estimation and optional inverse reconstruction on an included observed sample or user inputs. `pipeline/` additionally contains the original source/instrument generation, response/TDI/conditioning and study drivers. Recipes stage explicit external files into fresh workspaces. One complete observed-TDI demonstration passed; the high-rate simulations and full-year studies have not been freshly repeated.

The observed-TDI demonstration executes the K256 inverse and returns finite channel series with the original cadence/count. Its search maps, masks, ranking and response agree exactly with the original study. The new short-record reconstruction was not compared with injected truth and is not a reproduction of Figure7. Finite transform errors and event-mask coverage matter; do not assume exact round-trip recovery. The saved reconstruction plotting examples still read stored curves.

## Scientific domains and conventions

- Figures 2–3 come from the separate five-amplitude MBHB study. Signal strength is not a calibrated false-alarm probability, and complex coherence is not used in the published ranking statistic.
- Figures 4–6 use separate two-day injections sharing one continuous noise realization. They are not independent noise trials or one combined record of all sources. Search time is absolute days 4.25–5.75; plotted time is relative to day 4. The saved case identifiers retain older descriptive epoch strings: the actual parameters in the manifest, not those strings, determine the injection epoch.
- WDM search: M=512, K=32, beta order=6, precision=6, 4-second sampling for these response studies. The search band is 1e-5 to 0.02 Hz. Generic API defaults instead allow up to0.1Hz and8events; use the supplied manifest settings (0.02Hz and30events) to reproduce this study. Selection uses seed/growth quantiles 0.99/0.95 and the remaining explicit settings in `data/analysis/manifest.json`.
- Search coefficients are normalized by the observed per-frequency median power. For each of six monitor channels, the background is the frequency-wise median observed power outside the union of all retained event masks. Event power is `max(P/B - 1, 0)`. The ratio sums the three zeta contributions over the three Sagnac contributions. Clipping does not remove all noise fluctuations. Invalid backgrounds and zero denominators are flagged.
- Stored `native` and `matched` identifiers mean reference and loud-glitch levels. Loud levels were chosen using injected A/E power; that separate simulation calibration is not a measured detection SNR and is not used to estimate the operational response background.
- Reciprocal simultaneous 12+21 and all-six-link disturbances can have low ratios despite their instrumental origin. The reference six-link leader and some weak reference leaders need not describe the injected disturbance. Candidate ranking is not physical source classification.
- Figure 7 displays only +/-0.008 day around source peaks. Its published numerical comparisons concern full series. Figures 8 and 10 use expanded reconstruction windows for their reported metrics. Figure 10 shows narrower +/-0.012-day merger windows; its reference is the aggregate injected MBHB contribution, not an isolated source waveform.
- Reconstruction uses unwhitened channel coefficients, K=256, both quadratures, and their arithmetic-mean synthesis without fitted amplitude or time shifts. Frequency-fluctuation simulations use Hz; Sangria uses dimensionless fractional frequency at 5-second cadence.
- Figure 9 includes all155 candidates; six MBHB associations use the unblinded catalog. The other149 candidates are not classified by this plot.

`data/reconstruction/metrics.json` keeps stored metrics separate from compact display samples, with their comparison domains. It must not be interpreted as a recomputation of the full-domain metrics from those cropped samples.

## Environment and provenance

`requirements-figures.txt` records the installed versions used for the portable plotting check; `requirements-analysis.txt` adds transform dependencies. These are not claims about all historical generation environments, and no fresh dependency installation was performed during packaging. NumPy/pandas suffice for the saved-map regression; WDM is loaded only when a transform is requested.

Input files have SHA256 checksums in `data/checksums.json`. Run `python examples/verify_data.py` to check them. Source-product hashes and extraction descriptions accompany the response/reconstruction inputs. Relative historical artifact identifiers describe provenance only; they are never runtime dependencies.

## Full-data reruns and future archive

The public Sangria release is linked through the [LISA Data Challenges](https://lisa-ldc.lal.in2p3.fr/) and cited in the paper. This repository does not automatically download it or assert redistribution permission for an entire challenge archive. Repeating the full-year search requires the exact release, channel conventions, 30-day chunk boundaries and study configuration.

A future full-data archive should identify exact input checksums, ephemerides, generator versions, effective source parameters, random seeds and filtering/cropping prescriptions. In particular, the original aligned-spin wrapper's effective inclination was zero; a nominal keyword should not be silently substituted as a new source configuration. Avoid substituting historical day-wise noise-generation products for the continuous-noise response-study inputs.

The study-specific generation/conditioning drivers are now archived with exact source hashes and an isolated launcher in `pipeline/`. Their large prerequisites remain external; `pipeline/INPUT_MANIFEST.json` records101 required file identities. Full production reruns, new injection populations, detection calibration and new scientific claims remain outside this packaging task.
