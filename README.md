# cWB-space

Code, data and examples accompanying **cWB-space: A time-frequency transient-search pipeline for LISA**, [arXiv:2609.12986v1](https://arxiv.org/abs/2609.12986v1), by Shubhanshu Tiwari, Yumeng Xu and Giovanni A. Prodi.

No project software or data license has yet been selected. See [NOTICE.md](NOTICE.md) before redistribution.

This repository provides:

- Portable examples that regenerate the paper's nine numerical figures from small saved scientific products. Figure 1 is supplied as its original diagram PDF.
- A runnable observed-TDI pipeline: WDM transforms, candidate selection/ranking, observed-background response ratios and optional inverse reconstruction.
- The original simulation, LISA response/TDI, conditioning and study-specific reconstruction drivers, with an isolated-workspace launcher and explicit external inputs.
- A runnable regression example that repeats candidate selection, ranking and the response calculation from saved observed WDM maps, with tests against the original outputs.
- Input checksums, injection parameters, figure provenance and the scope of each comparison.

AI-assisted validation and plotting scripts are collected in [`pipeline/research/gpt6_astra_validation_scripts/`](pipeline/research/gpt6_astra_validation_scripts/). The repository documentation was prepared with assistance from OpenAI Codex (GPT-6 Astra), under the authors’ supervision.

## Reproduce the figures

Use Python 3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python examples/reproduce_all.py
```

PDFs and PNGs are written to `outputs/`. To choose another location:

```bash
python examples/reproduce_all.py --output-dir /path/to/figure-output
```

No input downloads, source simulations, WDM transforms or waveform inversions occur in the plotting examples. Reconstruction plots use saved reconstructed samples. This reproduces the published visualizations; it is not a fresh end-to-end regeneration of every scientific result.

| Paper figure | Subject | Example |
|---|---|---|
| 1 | Pipeline overview | Original artwork in `assets/figure01_pipeline.pdf` |
| 2–3 | Signal strength and A/E coherence | `examples/plot_strength.py` |
| 4–6 | MBHB, burst and multi-link response studies | `examples/plot_response.py` |
| 7–8, 10 | Aligned-spin, precessing and Sangria reconstructions | `examples/plot_reconstruction.py` |
| 9 | All 155 Sangria search candidates | `examples/plot_sangria.py` |

Every plotting example accepts `--output-dir`. Historical output stems such as `fig07_mbhb_snr_rho_event` refer to earlier working figure numbering; the table above gives the arXiv numbering.

## Run the pipeline on observed data

```bash
python -m pip install -r requirements-analysis.txt
python examples/run_pipeline.py --output-dir outputs/tdi-demo --response --inverse
```

This executes the transforms and analysis on the included 3 MB observed-TDI example. It reproduced the paper-study search maps, event mask, ranking and response ratio exactly; the optional inverse is a selected follow-up demonstration. Use `--input` for your own data and `--help` for time/channel keys and search settings. Defaults describe this example, not arbitrary LISA records.

For the **original simulation and paper-study workflows**:

```bash
python pipeline/run.py list
python pipeline/run.py plan continuous-noise
```

Follow the [pipeline guide](pipeline/README.md) to supply external inputs, prepare a fresh workspace and execute a stage. The source tree includes instrument/source generation, LISA response, TDI, conditioning, precessing-binary studies and Sangria search/reconstruction. Full production suites have not been rerun during packaging.

## Run the saved-map regression example

```bash
python examples/recover_candidates.py
python -m unittest discover -s tests -v
python examples/verify_data.py
```

This repeats the actual clustering and ranking on an included observed map and estimates the monitor-channel background outside the union of selected event pixels. It uses no separately supplied noise realization. The reference case's response ratio is approximately **0.016540485**. Regression tests also repeat the reconstruction-window selection on a saved observed map.

The search/transform code is in `cwb_space/blind_event_clustering.py` and `cwb_space/clean_wdm.py`. `response.py` provides `observed_background_ratio`; `reconstruction.py` contains the connected-window selection and two-quadrature inverse convention. To use WDM transforms on your own channel data, install the additional dependencies:

```bash
python -m pip install -r requirements-analysis.txt
python -m pip install --no-deps --no-build-isolation -e .
```

The latter installs the local code for import from other projects. It does not install the example data as wheel resources; run the reproduction examples from a checkout.

See [the reproducibility guide](docs/REPRODUCIBILITY.md) for data domains, conventions, limitations, and what a full simulation rerun would additionally require. No detection threshold or calibrated false-alarm probability is supplied by these examples; a small response ratio alone does not identify a gravitational wave.

## Layout

```text
cwb_space/       Search, response and reconstruction primitives
examples/        Portable plotting and saved-map analysis examples
data/            Scientific inputs, parameters and provenance
assets/          Original pipeline diagram
pipeline/        Original study drivers, configurations and isolated launcher
tests/          Regression checks against stored research results
docs/           Reproducibility and release-review notes
```

Large instrument/orbit products, historical source/noise checkpoints and full-year Sangria input remain external. The relevant producing and analysis code is now included; workflow recipes explicitly list required files and record their hashes. Exploratory outputs and manuscript drafts are not bundled.

Use [CITATION.cff](CITATION.cff) to cite the accompanying paper.
