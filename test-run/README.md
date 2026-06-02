# Noisy SMBHB Multi-Resolution WDM + Kappa Test Run

This folder contains two fixed single-SMBHB full-noise six-link simulations and noisy multi-resolution WDM+kappa analyses.

## Sources

- `smbhb_m1e6`: total mass `1e6 Msun`, RA `0.70`, Dec `-0.55`
- `smbhb_m1e7`: total mass `1e7 Msun`, RA `3.80`, Dec `0.62`

Both simulations generated full noisy six-link products with `lisainstrument`:

- `outputs/six_links_gw_only.csv`
- `outputs/six_links_noisy.csv`
- `outputs/lisa_instrument_output.h5`
- `outputs/tdi_michelson_XYZ_AET.csv`
- `outputs/tdi_gw_only_XYZ_AET.csv`

The final kappa analysis was run on `--source noisy`, not `gw_only`. WDM pixel selection can now use proper pyTDI A/E products while the kappa scan still uses the six-link Sagnac construction.

## Commands Used

Initial full-noise run was started through `cluster.cli`; it was stopped after `smbhb_m1e6` produced complete noisy six links because the remaining TDI stage is not used by WDM+kappa.

The six-link-only helper was used for `smbhb_m1e7`:

```bash
conda run -n lisa-tf python test-run/run_noisy_six_link_only.py \
  --config test-run/config.yaml \
  --catalog test-run/catalog.csv \
  --runs-root test-run/simulations \
  --source-id smbhb_m1e7
```

pyTDI A/E was then generated from the existing HDF5 products without rerunning `lisainstrument`:

```bash
conda run -n lisa-tf python test-run/generate_tdi_from_existing_h5.py \
  --config test-run/config.yaml \
  --catalog test-run/catalog.csv \
  --runs-root test-run/simulations
```

Noisy multi-resolution WDM+kappa command pattern:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir test-run/simulations/<source_id> \
  --output-dir test-run/coronagraphic/<source_id>_noisy \
  --source noisy \
  --nside 4 \
  --chunk-duration-s 32768 \
  --wdm-ms 128,256,512 \
  --top-pixel-fraction 0.10 \
  --min-cluster-size 5
```

pyTDI-AET WDM selection plus separate six-link kappa scan:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir test-run/simulations/<source_id> \
  --output-dir test-run/coronagraphic/<source_id>_noisy_pytdi_aet_offsource_wdmnorm \
  --source noisy \
  --nside 4 \
  --chunk-duration-s 32768 \
  --wdm-ms 128,256,512 \
  --top-pixel-fraction 0.10 \
  --min-cluster-size 5 \
  --wdm-aet-source pytdi \
  --selection-normalization offsource-frequency-median \
  --offsource-edge-fraction 0.20
```

Off-source WDM-background-normalized command pattern:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir test-run/simulations/<source_id> \
  --output-dir test-run/coronagraphic/<source_id>_noisy_offsource_wdmnorm \
  --source noisy \
  --nside 4 \
  --chunk-duration-s 32768 \
  --wdm-ms 128,256,512 \
  --top-pixel-fraction 0.10 \
  --min-cluster-size 5 \
  --selection-normalization offsource-frequency-median \
  --offsource-edge-fraction 0.20
```

This ranks WDM pixels by `AE power / off-source frequency-row median` before kappa scanning. The off-source estimate uses the first and last 20% of WDM time bins in the selected chunk.

Overlay command pattern:

```bash
conda run -n lisa-tf python choronagraphic-cluster/plot_multires_wdm_overlay.py \
  --run-dir test-run/coronagraphic/<source_id>_noisy \
  --background-m 512
```

Cross-resolution merge command pattern:

```bash
conda run -n lisa-tf python -m scripts.coronagraphic_multires_merge \
  --run-dir test-run/coronagraphic/<source_id>_noisy \
  --min-cluster-size 2 \
  --min-resolutions 2
```

## Results

See `test-run/summary.csv` for compact counts.
See `test-run/summary_raw_vs_wdmnorm.csv` for the raw-vs-normalized comparison.
See `test-run/summary_aet_source_comparison.csv` for internal-A/E versus pyTDI-A/E WDM selection.

Important caveat: the two noisy runs produced nearly identical cluster counts and event structures. This indicates the top-10% noisy-pixel selection is dominated by instrumental noise structure in this configuration, not by the difference between the `1e6` and `1e7` injected SMBHBs. A cleaner detection comparison should also run the same kappa analysis on `gw_only`, or apply whitening/noise suppression before top-pixel selection.

After WDM background normalization, the largest raw noisy event is split into multiple smaller cross-resolution event candidates. The two source masses still remain very similar, so this is an improvement in pixel ranking, not yet a validated SMBHB detector.

Using pyTDI A/E for WDM selection makes the two masses diverge more clearly: `smbhb_m1e6` produces 8 merged event candidates with largest event size 697 pixels, while `smbhb_m1e7` produces 6 merged event candidates with largest event size 1112 pixels. Kappa is still computed separately from the six-link Sagnac data.
