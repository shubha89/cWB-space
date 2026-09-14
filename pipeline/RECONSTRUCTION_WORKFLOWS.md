# Original strength, reconstruction and Sangria workflows

The registry `recipes/reconstruction.json` identifies the original drivers and their exact staged input paths. This source snapshot supplies actual waveform/response/search/reconstruction code, not only figure data. Packaging these drivers does not claim that every production run was repeated on this release environment. Run them only through the fresh-workspace launcher, never directly in the immutable research snapshot.

## Stage chains

- Figures2/3: `strength-study` reads the original two-week source/noise product and MBHB catalog, runs the fixed five-strength study, and creates scalar CSV products. The archived paper figure driver has `make_mbhb_snr_event_statistic` and `make_ae_coherence_mbhb_noise` functions for rendering those regenerated products. Its full main also creates unrelated figures.
- Figure7: `aligned-reconstruction` → `plot-aligned-original`. This retains M512/K256, .1Hz band, catalog-informed windows and unfitted observed-cluster output. Optional `aligned-window-check` is a separate subsequent check, not a replacement Figure7 mask.
- Figure8: `precessing-original-3` → `precessing-earlier-3`; independently run `precessing-finite-1` and `precessing-finite-2`; then `precessing-window-growth` → `plot-precessing-original`. The final case03 driver needs the original case03 polarization array and report for a common-time source comparison and fixed-old-domain metric, so the historical initialization stage remains explicit.
- Figure9: `sangria-full-search` → `plot-sangria-candidates-original`. Plotting additionally needs the separately verified six-association table; search rank alone does not determine source identity.
- Figure10: full-year candidate CSV → `sangria-k256-03`, `sangria-k256-08`, `sangria-k256-09`, `sangria-k256-11` → `sangria-window-growth` → `plot-sangria-reconstruction-original`. The mask originates in K32 search; K256 uses exact grid equality, raw coefficients and two-quadrature synthesis. The response reference is the aggregate MBHB-only TDI, not an isolated source.

A recipe can start from regenerated prior-stage output or separately staged immutable historical output. Record which route was used; staging an existing reconstruction is not rerunning its producing stage. Numerical and plotting recipes have separate names. The compact `examples/` plotters continue to use bundled cropped data; they do not automatically switch to newly regenerated workflow outputs.

## External input roles

Supply these exact role paths within a fresh workspace, with verified checksums; the launcher must not download them implicitly:

- Two-week checkpoint: `diagnostics/simulated_2week_mbhb_then_glitch_highrate/outputs/two_week_mbhb_then_glitch_tdi.npz` and `mbhb_catalog.csv` for the strength/aligned studies.
- Precessing response additionally requires `inputs/orbits_2week.h5`, plus `outputs/mbhb_week1_sagnac_zeta_products.npz` and `outputs/glitch_week2_sagnac_zeta_products.npz` under that same checkpoint directory. Imported helpers open these even when only A/E curves are eventually displayed.
- Full-year search: `data/LDC2_sangria_blind_v2.h5`, with structured `obs/tdi` fields t,X,Y,Z and dt attribute. Original 30-day chunk search assumes five-second data and .5day guards.
- Targeted Sangria: `data/mbhb-unblinded.h5` containing both `obs/tdi` and `sky/mbhb/tdi` with t,X,Y,Z fields. The latter is aggregate source response, not the catalog table. Both HDF observation roles were previously checked identical; do not infer equivalence from their names.
- Figure9/10 identity annotations: `.paper_agent/validation/sangria_top_v1/top_candidate_associations.csv`, a separately verified six-row table. Its construction is not an automatic output of the blind search.
- Intermediate generated artifacts are listed explicitly per recipe; never link writable historical output directories wholesale.

The original drivers compute root paths from their file depth; keep all archived paths unchanged beneath each staged workspace and run with workspace CWD. Some outputs are hard-coded and some directories must not exist before execution. The Sangria growth driver performs numerical work at import; recipe listing and dependency checks must read the registry without importing it. The aligned-window diagnostic writes beside its own copied script, which must therefore be in the writable workspace. Prepare only explicitly required plot-parent directories; do not precreate fresh-output scientific directories.

## Environment and validation limits

The precessing source uses the original `phenomxpy`0.1.1 implementation. Newer source-generator versions may change waveform conventions; verify effective parameters rather than silently upgrading. WDM research usage is M512/K256/beta6/precision6/numba for reconstruction, M512/K32 for search. Historical module initialization also imports instrument/orbit/glitch utilities through `scripts/__init__.py`, so the full dependency closure is larger than the saved-map example. NumPy, pandas, SciPy, Matplotlib, h5py, wdm-wavelet, numba, phenomxpy, lisagwresponse, PyTDI, LISA Orbits, LISA Instrument and LISA Glitch are required for the broad research tree. The Sangria K256 report also queries installed `jax` metadata despite selecting numba; verify distribution availability before launching.

No full numerical run was executed to package these recipes. Static source identity and dependency checks, then a fresh-workspace dry run, are appropriate first checks. Validate a bounded original-setting stage before spending resources on full-year or high-rate regeneration. The source initialization comparison, inverse residuals, finite-record conditioning, original mask equality and unfavorable reconstruction outcomes must not be removed to make a run pass. Full input redistribution and software licensing remain separate author decisions.
