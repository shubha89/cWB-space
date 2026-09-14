# Pipeline used for the paper

This directory contains the actual simulation, response, TDI, conditioning, search and reconstruction drivers used for the paper studies, with their local import dependencies and frozen source configurations. The files under `research/` preserve the original algorithms and configurations, with the validation directory renamed to `gpt6_astra_validation_scripts`. They are not a new scientific implementation.

The `gpt6_astra_validation_scripts/` directory contains the AI-assisted validation and plotting scripts. Directory references were updated consistently; original and packaged source hashes are retained in `SOURCE_MANIFEST.json`. Only relevant drivers and configuration files are included, not private conversation logs or research notes. The launcher supplies writable workspaces and explicit input files so these paths no longer depend on the original author's checkout. Do not run the archived drivers directly in the source snapshot: some perform work immediately when imported, and some reset their output directories.

For a portable observed-TDI search and follow-up, start instead with:

```bash
python -m pip install -r requirements-analysis.txt
python examples/run_pipeline.py --output-dir outputs/tdi-demo --response --inverse
```

This runs the WDM transforms on the included raw observed sample, selects and ranks candidates, estimates the response ratio from observed monitor channels, and reconstructs the selected event. It uses no injected waveform in selection. The sample, conventions and validation are described in [data/pipeline/README.md](../data/pipeline/README.md). The full scientific-study drivers below reproduce the original study-specific source generation, known-window controls and Sangria analyses.

## Original study launcher

From the repository root:

```bash
python -m pip install -r requirements-pipeline.txt
python pipeline/run.py list
python pipeline/run.py plan continuous-noise
```

`plan` is read-only. It reports the entrypoint, exact required input paths, output paths, dependency availability and expected resources. It does not import the research drivers, download data or start a simulation.

Supply the external files using the recipe paths below an input root, or use a JSON mapping from each historical relative path to its actual location:

```bash
python pipeline/run.py plan continuous-noise --external-data /path/to/paper-inputs
python pipeline/run.py prepare continuous-noise \
  --external-data /path/to/paper-inputs --workspace /path/to/new-noise-workspace
python pipeline/run.py run --workspace /path/to/new-noise-workspace --timeout-seconds 1800
```

`prepare` requires a new directory. It copies the archived code and only the declared external inputs, never links writable original outputs. Copied files are hashed and compared with the historical input manifest where a reference exists. A differing regenerated input is flagged in the receipt, not silently treated as the historical file. `run` is the only command that executes a driver. It writes stdout/stderr logs and a `pipeline-run.json` receipt, checks input integrity, records the exit status and checks expected outputs. The receipt state `COMPLETED` means successful exit, expected output presence and input integrity; it does not certify scientific agreement. A workspace isolates file layout but is not an operating-system sandbox. A failed or completed workspace cannot be rerun accidentally; prepare a new one. Dependency installation is always a separate user action.

A file map can combine outputs from different stages without reorganizing the original data:

```json
{
  "gpt6_astra_validation_scripts/validation/continuous_noise_control_v1/noise_continuous.npz": "/path/to/noise-workspace/gpt6_astra_validation_scripts/validation/continuous_noise_control_v1/noise_continuous.npz",
  "gpt6_astra_validation_scripts/validation/continuous_noise_control_v1/noise_split.npz": "/path/to/noise-workspace/gpt6_astra_validation_scripts/validation/continuous_noise_control_v1/noise_split.npz"
}
```

Pass it as `--input-map paths.json`; combine with `--external-data` for other inputs. Later stages may use regenerated prior-stage products or historical inputs; the receipt records which files were supplied. Dependencies listed in recipes describe ordering, not automatic execution. The launcher does not recursively launch expensive prerequisite stages.

For `response-analysis`, the copied source manifest's `source_path` fields are rebased to the copied source files. Only filesystem locations change; all scientific parameters stay unchanged. The receipt records every path relocation and original/prepared hashes.

## Study coverage

| Results | Workflow |
|---|---|
| Figures 2–3 | `strength-study`; plot its CSV directory with `examples/plot_strength.py --data-dir /path/to/study/outputs --output-dir outputs/strength` |
| Figures 4–6 | `continuous-noise`, `clean-sources`, `continuous-analysis`, `response-analysis`, then `plot-response-original` and `plot-multilink-original` |
| Figure 7 | `aligned-reconstruction` → `plot-aligned-original` |
| Figure 8 | Precessing finite-observation/earlier-start stages → `precessing-window-growth` → `plot-precessing-original` |
| Figure 9 | `sangria-full-search` → separately verified source associations → `plot-sangria-candidates-original` |
| Figure 10 | Four `sangria-k256-*` stages → `sangria-window-growth` → `plot-sangria-reconstruction-original` |

See [RECONSTRUCTION_WORKFLOWS.md](RECONSTRUCTION_WORKFLOWS.md) for exact ordering and configuration distinctions. The simple compact plotting examples still use their bundled samples; they do not automatically adopt a new production run's outputs.

## External data and numerical limits

Large orbit, noise, source-response and Sangria files are external. `INPUT_MANIFEST.json` records the original available input hashes and sizes; `SOURCE_MANIFEST.json` records the archived source hashes. Recipes enumerate every supplied role. The original continuous source generator reuses one cached single-link response; its orbit and historical time-axis inputs are also explicit dependencies. The original continuous-analysis control additionally consumes two cached source responses. These prerequisites are not generated by the launcher or replaced with different injections.

The high-rate noise and full source suites can take tens of minutes and multiple gigabytes. Resource figures in recipes are historical measurements or planning estimates, not enforced portable memory/disk caps. `--timeout-seconds` limits the direct driver wall time; original driver assertions are retained, including platform-dependent `ru_maxrss` conventions. Check capacity before production execution.

Packaging checks established exact source copies, static recipe/input coverage, a successful staged original plotting run, and an actual observed-TDI search/response/inverse demonstration. The high-rate instrument simulation, all41 response trials, all precessing generations and the full-year scan were not newly rerun during packaging. Exact recreation on another environment still needs validation against the recorded products and effective parameters. No outcome should be repaired by silently changing source parameters or selection rules.

Existing external archives using the former `.paper_agent/` directory are accepted as an input-only fallback. Prepared workspaces and new outputs use `gpt6_astra_validation_scripts/`.
