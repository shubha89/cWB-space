# Release validation and remaining limitations

Updated 14 September 2026. Repository integration, commit and push authorized by the principal investigator.

The paper pipeline and examples are integrated into the existing LISA-TF repository. Earlier research files are retained locally outside the tracked release file set, and the existing Git history is preserved.

## Completed checks

- All10 reference figure PDFs match the public arXiv v1 source archive byte-for-byte.
- Nine numerical figures regenerate successfully; Figure1 is copied from the supplied artwork. All10 resulting PDFs render pixel-identically to the arXiv originals at90dpi using the same local renderer.
- Scientific regression tests pass: original search masks/all event scalars and response sums; background exclusion/undefined denominator handling; original expanded reconstruction mask.
- The plotting pipeline, analysis example and regression tests run from a separate copied checkout while the working directory is outside the research repository and PYTHONPATH is unset. No private working-directory imports are required.
- Bundled input checksums match. Archived drivers retain their study layout with the validation directory renamed to `gpt6_astra_validation_scripts`; the launcher stages it in a fresh workspace with explicit copied inputs. A historical temporary-cache fallback is overridden by the launcher. No personal workstation path or vendored dependency is required.
- Generated figures/caches/large external inputs are ignored by Git. Python test files and small scientific NPZ/CSV files are included, unlike the old checkout's broad ignore rules.
- An actual observed-TDI search/response/inverse run completed in14seconds using502MiB. Maps, masks, all candidate values and response sums/ratio matched the original study exactly. The optional inverse completed with finite samples; this is an operational follow-up check, not a new truth-overlap validation.
- Original drivers (47files) and24workflow recipes are included. All101declared external inputs and required modules are present in the research environment; no full production simulation was launched. One original plotting stage ran through the isolated-workspace launcher with unchanged copied inputs.
- No dependency installation was performed. Pins describe the inspected environment, not a separately verified fresh installation.

## Remaining release limitations

1. Select software and data licenses and confirm contributor/redistribution permissions. WDM-wavelet's GPLv3 dependency warrants compatibility review; no project license has been invented here. The arXiv paper license is not a blanket license for software and data.
2. Review the included full study code and workflow recipes. Select the large-data archive/distribution scope and any full production reruns required before release. Source/response/TDI/simulation code is now included; the large prerequisite files remain external.
3. Supply the editable Figure1 artwork if available; otherwise retain its supplied PDF and the explicit reproduction limitation.
4. Hosting is the existing `shubha89/LISA-TF` repository; integration and publication were explicitly authorized on 14 September 2026. No software/data license is selected by that authorization.

Full reconstruction-domain metrics are stored for reference; the cropped plotted waveform windows cannot independently recompute all of them. The observed-TDI inverse has been executed successfully, but astrophysical recovery for that short-record follow-up was not newly validated against truth. These limits are also stated in the public-facing reproducibility guide.
