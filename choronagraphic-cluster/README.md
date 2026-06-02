# Choronagraphic Pixel Clustering Prototype

This prototype tests a physics-guided clustering idea on one SMBHB:

1. Load one simulated SMBHB six-link data product.
2. Build A/E WDM maps and select high-power time-frequency pixels.
3. Build first-generation Sagnac generators from the same links.
4. For every selected WDM pixel, scan coronagraphic `kappa` over a HEALPix sky grid.
5. Assign each pixel to its best coronagraphic-null sky direction.
6. Cluster pixels using compact sky-response features.

The folder name follows the requested spelling, `choronagraphic-cluster/`. The implementation calls the existing `coronagraphic_tdi.py` module.

## First Smoke Run

Use a small HEALPix grid first:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir cluster/runs/simulations/smbhb_00000 \
  --output-dir choronagraphic-cluster/runs/smbhb_00000_smoke \
  --source gw_only \
  --nside 2 \
  --chunk-duration-s 8192 \
  --wdm-m 256 \
  --top-pixels 20
```

Then scale:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir cluster/runs/simulations/smbhb_00000 \
  --output-dir choronagraphic-cluster/runs/smbhb_00000 \
  --source gw_only \
  --nside 4 \
  --chunk-duration-s 32768 \
  --wdm-m 512 \
  --top-pixels 80
```

## Multi-Resolution Run

This mode scans several WDM resolutions. For each WDM layer, it:

1. Builds the A/E WDM map at that resolution.
2. Selects the top fraction of valid pixels in the requested frequency band.
3. Computes coronagraphic `kappa` over the HEALPix sky grid.
4. Clusters pixels independently for that WDM resolution.

Small validation run:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir cluster/runs/simulations/smbhb_00000 \
  --output-dir choronagraphic-cluster/runs/smbhb_00000_multires_test \
  --source gw_only \
  --nside 2 \
  --chunk-duration-s 8192 \
  --wdm-ms 128,256,512 \
  --top-pixel-fraction 0.10
```

Fuller prototype run:

```bash
conda run -n lisa-tf python choronagraphic-cluster/run_one_smbhb.py \
  --work-dir cluster/runs/simulations/smbhb_00000 \
  --output-dir choronagraphic-cluster/runs/smbhb_00000_multires \
  --source gw_only \
  --nside 4 \
  --chunk-duration-s 32768 \
  --wdm-ms 128,256,512 \
  --top-pixel-fraction 0.10
```

## Outputs

Each run writes:

- `candidate_pixels.csv`: selected A/E WDM pixels.
- `healpix_sky_grid.csv`: scanned sky directions.
- `pixel_clusters.csv`: per-pixel best sky direction and HDBSCAN label.
- `kappa_sky_responses.npz`: raw response matrix `[candidate_pixel, sky_direction]`.
- `summary.json`: run summary.
- `plots/`: diagnostic plots when plots are enabled.

For multi-resolution runs, each layer is written under:

- `wdm_m_<M>/candidate_pixels.csv`
- `wdm_m_<M>/pixel_clusters.csv`
- `wdm_m_<M>/kappa_sky_responses.npz`
- `wdm_m_<M>/summary.json`

The top-level output directory also contains:

- `candidate_pixels_all_resolutions.csv`
- `pixel_clusters_all_resolutions.csv`
- `healpix_sky_grid.csv`
- `summary.json`

To overlay all multi-resolution coherent pixels on one WDM power map:

```bash
conda run -n lisa-tf python choronagraphic-cluster/plot_multires_wdm_overlay.py \
  --run-dir choronagraphic-cluster/runs/smbhb_00000_multires_test \
  --background-m 512
```

This writes:

- `plots/multires_selected_overlay_on_wdm.png`
- `plots/multires_coherent_overlay_on_wdm.png`

To merge coherent clusters across WDM resolutions with a second HDBSCAN pass:

```bash
conda run -n lisa-tf python -m scripts.coronagraphic_multires_merge \
  --run-dir choronagraphic-cluster/runs/smbhb_00000_multires_test \
  --min-cluster-size 2 \
  --min-resolutions 2
```

This writes:

- `layer_cluster_representatives.csv`: one summary row per coherent per-resolution cluster.
- `merged_pixel_clusters.csv`: original pixels plus cross-resolution `event_cluster` labels.
- `multires_event_summary.csv`: merged event candidates and WDM-resolution support.
- `multires_merge_summary.json`: compact merge diagnostics.
- `plots/multires_event_clusters_time_frequency.png`
- `plots/multires_event_clusters_sky.png`
- `plots/multires_event_resolution_support.png`

## Interpretation

For each selected WDM pixel, the important columns are:

- `best_ra`, `best_dec`: sky direction where `kappa` is most suppressed.
- `sky_contrast`: median sky response divided by the best-null response.
- `sky_sharpness`: 10th-percentile sky response divided by the best-null response.
- `cluster`: HDBSCAN cluster label. `-1` means noise/unclustered.
- `true_sky_sep_deg`: angular separation from the injected source sky position when metadata are available.

This is a prototype. It uses static equilateral LISA geometry even though the simulation used realistic orbits. The first success criterion is not perfect localization; it is whether pixels from the same SMBHB track prefer a coherent sky patch.
