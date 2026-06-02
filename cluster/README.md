# SMBHB WDM Clustering

This folder contains the first A/E WDM clustering pipeline:

1. Sample SMBHB parameters over `M_total = 1e4..1e8` and `eta = 0.15..0.25`.
2. Generate `phenomxpy` SMBHB waveforms.
3. Run the repo's existing LISA link and TDI simulation pipeline.
4. Estimate an off-source A/E noise PSD from one reference simulation.
5. Whiten A/E time series, convert them to WDM maps, and crop fixed-size A/E patches.
6. Train a CNN autoencoder.
7. Cluster latent vectors with HDBSCAN.
8. Evaluate clusters against physical metadata.

The repo uses folder labels such as `eta-1_5` for `eta=0.15`. The config accepts the physical value `0.15`; the sampler smoke check also accepts label-style bounds such as `1.5` and converts them to `0.15`.

## Environment

Update the conda environment after this change:

```bash
conda env update -n lisa-tf -f env.yml
```

The existing `lisa-tf` environment already has `phenomxpy` and `wdm-wavelet` here, but the autoencoder and HDBSCAN stages need the new ML dependencies.

## Minimal Workflow

```bash
conda run -n lisa-tf python -m cluster.cli sample --n-sources 100
conda run -n lisa-tf python -m cluster.cli simulate --limit 10
conda run -n lisa-tf python -m cluster.cli estimate-psd --reference-work-dir lisa_link_sim/eta-1_5
conda run -n lisa-tf python -m cluster.cli build-dataset
conda run -n lisa-tf python -m cluster.cli train-autoencoder
conda run -n lisa-tf python -m cluster.cli cluster
conda run -n lisa-tf python -m cluster.cli evaluate
```

Run a fast validation check:

```bash
conda run -n lisa-tf python -m cluster.cli smoke
```

The full `simulate` stage is expensive. Start with `--limit 2` or `--limit 10`, verify the dataset and clustering outputs, then scale up.

## Outputs

Default outputs are written under `cluster/runs/`:

- `catalog.csv`: sampled SMBHB parameters.
- `simulations/<source_id>/`: generated waveform and LISA simulation products.
- `noise_psd.csv`: off-source A/E PSD from the reference run.
- `wdm_patches.npz`: `[n, 2, freq, time]` A/E WDM patch tensor.
- `wdm_patches_metadata.csv`: source metadata for interpretation.
- `autoencoder.pt`: trained PyTorch autoencoder.
- `embeddings.npz`: latent vectors.
- `hdbscan_labels.csv`: cluster labels.
- `cluster_metrics.json` and `cluster_summary.csv`: evaluation outputs.
