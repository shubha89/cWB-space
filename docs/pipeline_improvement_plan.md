# cWB-Space Pipeline Improvement Plan

This document is a working brainstorming space for the next development stages of the cWB-inspired LISA pipeline. The goal is to turn rough ideas into a practical sequence of improvements, tests, and validation milestones.

## 1. Current Pipeline Snapshot

Briefly describe the current state before deciding what to improve.

- Input data products:
- TDI channels currently supported:
- Time-frequency transform currently used:
- Clustering method:
- Detection statistic:
- Reconstruction output:
- Validation datasets:
- Known limitations:

## 2. Development Goals

Primary goals to clarify:

- Improve burst sensitivity for LISA-like signals.
- Preserve the coherent cWB philosophy where useful.
- Adapt the method to space-based detector geometry, TDI channels, and non-stationary LISA noise.
- Make the pipeline easier to validate, reproduce, and compare against baselines.

Open questions:

- Is this pipeline LISA-specific, or should it be general for space-based detectors?
- Is the first target SMBHB bursts, generic unmodelled bursts, glitches, or multiple source classes?
- Should the pipeline prioritize detection, sky localization, waveform reconstruction, or all three?

## 3. Target Pipeline Modules

The final pipeline should be organized around a small number of clear modules. The detailed implementation can evolve later, but the module boundaries should guide the next development work.

### Module 1: Simulation Of Synthetic Data

Purpose:

- Generate synthetic LISA-like input data for development, testing, and validation.
- Include the waveform simulator, LISA response, orbit handling, and instrument simulator.
- Produce the six-link data products needed by downstream TDI computation.

Inputs:

- Source parameters.
- Waveform model configuration.
- LISA orbit and response configuration.
- Instrument/noise configuration.
- Sampling, duration, burn-in, and output-path configuration.

Outputs:

- Six GW-only link data.
- Six noisy instrument link data.
- Orbit and response metadata.
- Run metadata needed for reproducibility.

To-do:

- Define the canonical output format for six-link data.
- Separate source/waveform generation from instrument/noise generation where practical.
- Ensure synthetic datasets can be regenerated from config files.
- Add a small reference simulation for smoke tests.
- Track metadata: source parameters, random seed, sample rate, time origin, orbit file, response settings, and noise settings.

### Module 2: TDI Computation From Six-Link Data

Purpose:

- Convert six-link data into TDI observables using pyTDI as the preferred source of truth.
- Support both Michelson-to-A/E/T and Sagnac-to-coronagraphic paths.

Subpaths:

- Michelson path: six links -> pyTDI X/Y/Z -> A/E/T.
- Sagnac path: six links -> pyTDI alpha/beta/gamma -> coronagraphic quantities.

Inputs:

- Six-link data products from Module 1.
- Orbit/delay information.
- TDI generation choice.
- Interpolation and unit conventions.

Outputs:

- Michelson X/Y/Z.
- Orthogonal A/E/T.
- Sagnac alpha/beta/gamma.
- Coronagraphic kappa or other null/coherent combinations.
- Validation diagnostics comparing pyTDI and any local reference formulas.

To-do:

- Keep `scripts/tdi_computation.py` as the current pyTDI Michelson/AET source until a cleaner module replaces it.
- Promote pyTDI Sagnac from diagnostic scripts into a validated production path.
- Decide which pyTDI Sagnac observables are required: `ALPHA/BETA/GAMMA`, `ALPHA*_ETA/BETA*_ETA/GAMMA*_ETA`, or both.
- Validate sign, scaling, Fourier, time-delay, and generation conventions.
- Decide whether local static TDI formulas remain only as reference/validation code.
- Define standard file names and column names for Michelson, A/E/T, Sagnac, and coronagraphic outputs.

### Module 3: Coherence Definition

Purpose:

- Define what "coherent" means for this LISA/cWB-inspired pipeline.
- Translate multi-channel TDI information into signal-consistency measures.

Status:

- This needs further theoretical and algorithmic work.
- Details will be defined later.

Possible ingredients to revisit later:

- Coherent signal energy.
- Incoherent residual energy.
- Null-stream or near-null consistency.
- Sky-dependent response consistency.
- Polarization consistency.
- Time-frequency cluster coherence.
- Channel covariance and correlated noise.

To-do:

- Write a short mathematical note defining the first coherent statistic.
- Decide which TDI basis is used for the first coherent implementation.
- Decide whether coherence is evaluated per pixel, per cluster, or both.
- Define diagnostic plots that show why a cluster is coherent.
- Compare the chosen statistic against a simple incoherent excess-power baseline.

### Module 4: Detection Statistics And Signal Likelihood

Purpose:

- Rank candidate events and quantify the likelihood that a signal is present in the data.
- Convert coherent features and cluster properties into detection statistics.

Inputs:

- Time-frequency maps.
- Candidate clusters.
- Coherence outputs from Module 3.
- Background/noise estimates.
- Optional source or response priors.

Outputs:

- Event ranking statistic.
- Signal likelihood or likelihood-ratio proxy.
- False-alarm or background-calibrated significance estimate.
- Detection-efficiency and sensitivity summaries.

To-do:

- Define the first minimal detection statistic.
- Decide how likelihood is represented: explicit likelihood, likelihood ratio, ranking score, or Bayesian evidence proxy.
- Define background estimation strategy for LISA-like data.
- Add injection-recovery tests.
- Produce ROC/sensitivity curves once the statistic is stable.
- Store event-level summaries in a reproducible table format.

## 4. Proposed Improvement Tracks

### Track A: Data And Preprocessing

Potential improvements:

- Make robust TDI computation the first development priority.
- Use `pyTDI` as much as possible rather than maintaining independent duplicated TDI logic.
- Support and validate both Michelson responses and Sagnac responses.
- Standardize input format for simulated and real-like LISA data.
- Add clear support for TDI combinations such as X/Y/Z, A/E/T, and possibly null channels.
- Improve whitening and noise PSD estimation.
- Handle gaps, edge effects, and non-stationary noise explicitly.
- Add configurable chunking and overlap rules.

Deliverables:

- A tested pyTDI-based TDI computation layer.
- Validation examples for Michelson and Sagnac responses.
- Reproducible preprocessing script.
- Unit tests for channel transforms and whitening.
- Diagnostic plots for PSDs, whitened data, and residuals.

Current code audit:

- Production Michelson TDI is already pyTDI-based in `scripts/tdi_computation.py`.
- `scripts/tdi_computation.py` imports `Data` from `pytdi` and `pytdi.michelson`.
- The configured Michelson generation selects `pytdi.michelson.X2/Y2/Z2` for generation 2 and `pytdi.michelson.X1/Y1/Z1` for generation 1.
- Noisy TDI is computed from `Data.from_instrument(...)`; GW-only TDI is computed from `Data.from_gws(...)`.
- A/E/T are not computed by a separate pyTDI observable; they are currently local linear combinations of the pyTDI-generated X/Y/Z:
  `A = (Z - X) / sqrt(2)`, `E = (X - 2Y + Z) / sqrt(6)`, `T = (X + Y + Z) / sqrt(3)`.
- The output files `outputs/tdi_michelson_XYZ_AET.csv` and `outputs/tdi_gw_only_XYZ_AET.csv` are therefore pyTDI-derived Michelson/AET products.
- The coronagraphic clustering path explicitly loads these pyTDI-derived A/E products for WDM pixel selection in `choronagraphic-cluster/run_one_smbhb.py`.
- Sagnac is not yet a production pyTDI data product in the same sense as Michelson XYZ/AET.
- `scripts/pytdi_sagnac_kappa_example.py` is a diagnostic bridge that computes `pytdi.sagnac.ALPHA*/BETA*/GAMMA*` from `Data.from_instrument(...)` or `Data.from_gws(...)`.
- `coronagraphic_tdi.py` still contains local static-geometry first-generation Sagnac and Michelson builders from six-link data. These are useful for current kappa/projection experiments, but they are not pyTDI-based and should not silently become the source of truth.
- Current mixed state: WDM selection uses pyTDI A/E, while coronagraphic kappa construction still uses local static Sagnac formulas in the main clustering path.

Relevant files:

- `scripts/tdi_computation.py`: pyTDI Michelson production path.
- `test-run/generate_tdi_from_existing_h5.py`: post-processing wrapper that regenerates pyTDI XYZ/AET CSVs from existing HDF5 products.
- `choronagraphic-cluster/run_one_smbhb.py`: loads pyTDI A/E products for WDM selection, but builds Sagnac frequency series from local static formulas.
- `scripts/pytdi_sagnac_kappa_example.py`: pyTDI Sagnac diagnostic/comparison script.
- `coronagraphic_tdi.py`: local static Sagnac/Michelson/coronagraphic response implementation.

Implication for next development:

- Treat `scripts/tdi_computation.py` as the current source of pyTDI Michelson/AET products.
- Promote pyTDI Sagnac from diagnostic status to a validated production path only after convention, generation, scaling, and delay-model agreement are checked against the coronagraphic response model.
- Decide whether to keep local static TDI builders only as validation/reference tools, or replace downstream Sagnac usage with pyTDI-derived Sagnac observables.

### Track B: Time-Frequency Representation

Potential improvements:

- Compare WDM, wavelets, Q-transform-like tilings, and multiresolution transforms.
- Tune time-frequency resolution for LISA signal durations.
- Add energy maps per TDI channel and coherent combinations.
- Improve thresholding before clustering.

Deliverables:

- Transform comparison notebook or script.
- Metrics for injected signal energy recovery.
- Visual diagnostics for selected injections.

### Track C: Coherent Statistic

Potential improvements:

- Define a LISA-adapted coherent likelihood/statistic.
- Use TDI channel correlations correctly.
- Include noise covariance between channels if relevant.
- Explore null-stream vetoes using T or other near-null combinations.
- Separate coherent signal energy from incoherent residual/glitch energy.

Deliverables:

- Mathematical note defining the statistic.
- Implementation with tests on controlled synthetic signals.
- Comparison against simpler incoherent excess-power baselines.

### Track D: Clustering And Event Building

Potential improvements:

- Compare connected-component clustering, HDBSCAN, and cWB-like constrained clustering.
- Add multiresolution cluster merging.
- Track cluster morphology features.
- Improve ranking of clusters using coherent statistics and vetoes.

Deliverables:

- Configurable clustering module.
- Cluster summary table format.
- Cluster-level plots and event reports.

### Track E: LISA Response And Reconstruction

Potential improvements:

- Improve handling of LISA antenna response and time-varying geometry.
- Add sky-grid or likelihood-based localization.
- Estimate source direction, polarization consistency, and reconstructed waveform.
- Compare reconstruction quality across TDI bases.

Deliverables:

- Response-model interface.
- Injection-recovery validation set.
- Reconstruction error metrics.

### Track F: Background Estimation And Significance

Potential improvements:

- Define a background strategy suitable for LISA where terrestrial time slides do not directly apply.
- Use noise-only simulations, off-source regions, bootstrap methods, or synthetic phase randomization.
- Estimate false alarm probability/rate for cluster statistics.
- Track sensitivity versus threshold.

Deliverables:

- Background generation procedure.
- Detection threshold calibration.
- ROC/sensitivity curves.

### Track G: Validation And Benchmarks

Potential improvements:

- Build a small benchmark suite with known injections.
- Track detection efficiency, false alarms, localization error, waveform mismatch, and runtime.
- Add regression tests for previously working examples.
- Make results reproducible from config files.

Deliverables:

- Benchmark dataset manifest.
- Automated evaluation script.
- Versioned result tables and plots.

### Track H: Software Architecture

Potential improvements:

- Separate pipeline stages into reusable modules.
- Define config schemas for runs.
- Standardize outputs into structured files.
- Add logging, metadata, and provenance.
- Reduce notebook-only logic by moving core pipeline code into scripts/modules.

Deliverables:

- Clear package/module structure.
- Example configs.
- End-to-end command-line entry point.
- Minimal CI or smoke tests.

## 5. Suggested Development Phases

### Phase 1: Stabilize The Existing Pipeline

Purpose: make the current implementation reproducible before changing algorithms.

Tasks:

- Establish robust TDI computation using `pyTDI` wherever possible.
- Validate both Michelson and Sagnac response generation.
- Document current end-to-end flow.
- Freeze one reference dataset and one reference config.
- Add smoke test for a known injection.
- Save standard plots and metrics.

Exit criteria:

- A new run can reproduce the same qualitative result from a clean command, starting from validated TDI products.

### Phase 2: Improve Preprocessing And TF Maps

Purpose: make the input representation reliable.

Tasks:

- Validate TDI channel handling.
- Compare whitening approaches.
- Add multiresolution TF diagnostic plots.
- Decide the default TF representation.

Exit criteria:

- Signal energy is visible and stable across repeated runs and expected channel choices.

### Phase 3: Add LISA-Coherent Ranking

Purpose: move from excess-power clustering toward a genuinely coherent LISA statistic.

Tasks:

- Define coherent energy, residual energy, and null energy.
- Implement first version of the statistic.
- Compare ranking against current clustering output.
- Add injection/noise comparison plots.

Exit criteria:

- Injected signals rank above noise clusters more reliably than with the current statistic.

### Phase 4: Event Reconstruction And Localization

Purpose: extract physical information, not only detections.

Tasks:

- Add response-model interface.
- Run sky-grid or coarse localization.
- Reconstruct waveform in selected channels.
- Quantify reconstruction accuracy.

Exit criteria:

- Pipeline returns event-level parameters and reconstruction diagnostics.

### Phase 5: Background And Sensitivity Study

Purpose: make detection claims statistically meaningful.

Tasks:

- Choose background estimation method.
- Run noise-only trials.
- Calibrate thresholds.
- Produce sensitivity curves.

Exit criteria:

- Results include false-alarm estimates and injection efficiency.

## 6. Immediate Next Decisions

Questions to answer before implementing the next round:

1. What is the main target signal class for the first paper/prototype?
2. Which TDI basis should be the default for development: X/Y/Z or A/E/T?
3. What exact pyTDI interfaces should be treated as the source of truth for Michelson and Sagnac responses?
4. What is the reference dataset for regression testing?
5. What is the minimum viable coherent statistic for the first version?
6. Should the name be `cWB-Space`, `cWB-LISA`, or something more independent?

## 7. Running Idea Backlog

Add ideas here without worrying about order.

- Compare current local TDI code against pyTDI outputs before replacing or refactoring anything.
- Null-channel veto for instrumental artifacts.
- Adaptive TF tiling for long-duration chirping signals.
- HDBSCAN versus connected-component clustering comparison.
- Cluster morphology features for ranking.
- Configuration-driven end-to-end runs.
- Standard event report with plots and metadata.
- Injection campaign with controlled SNR and sky location grid.
- Sensitivity comparison against a non-coherent excess-power baseline.

## 8. Notes From Brainstorming

Use this section for meeting notes, rough equations, references, and unresolved design questions.
