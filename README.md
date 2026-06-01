# LISA-TF: Gravitational Wave Detection Analysis

Analysis and simulation code for LISA (Laser Interferometer Space Antenna) gravitational wave detection using time-delay interferometry (TDI).

## Project Structure

```
├── lisa_link_sim/              # Main simulation notebooks and data
│   ├── *.ipynb                # Jupyter notebooks for analysis
│   ├── inputs/                # Input data files
│   └── outputs/               # Generated output files
├── scripts/                    # Reusable importable pipeline scripts
│   ├── six_link_generation.py  # Waveform alignment, orbit, GW links, noisy links
│   ├── tdi_computation.py      # Noisy and GW-only XYZ/AET TDI products
│   ├── wdm_computation.py      # Multi-resolution WDM and T-weighted A/E maps
│   ├── plotting_utils.py       # Shared plotting helpers
│   ├── lisa_link_simulation.py # Compatibility wrapper for the main pipeline
│   └── tdi_wdm_pipeline.py     # Compatibility wrapper for WDM post-processing
├── examples/                   # Runnable examples and smoke checks
├── coronagraphic_tdi.py        # TDI analysis module
├── create_lisalink.py          # LISA link creation utilities
└── env.yml                     # Conda environment specification
```

## Setup Instructions

### Prerequisites
- Conda (Miniconda or Anaconda)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LISA-TF
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f env.yml
   conda activate lisa-tf
   ```

3. **Verify installation**
   ```bash
   python -c "import lisaorbits; import lisagwresponse; import lisainstrument; import pytdi; print('All packages imported successfully!')"
   ```

## Running the Analysis

1. Start Jupyter Lab
   ```bash
   jupyter lab
   ```

2. Navigate to `lisa_link_sim/` and open the relevant notebooks:
   - `orbit-lisa.ipynb` - Orbital analysis
   - `phenomxpy-waveform.ipynb` - Waveform generation
   - `TDI-test.ipynb` - TDI calculations
   - `lisa-link-with-noise.ipynb` - Link analysis with noise

### Reusable Pipeline Scripts

The reusable scripts live in `scripts/` and can be called from notebooks or Python:

- `scripts/six_link_generation.py` generates aligned waveform files, orbit files, six GW-only links, and six noisy links.
- `scripts/tdi_computation.py` computes noisy and GW-only XYZ/AET TDI from the six-link products.
- `scripts/wdm_computation.py` computes multi-resolution WDM products and noisy A/E maps weighted by the noisy T-channel proxy.
- `scripts/plotting_utils.py` contains shared plotting helpers for link, TDI, WDM, best-pixel, and zoomed views.
- `scripts/lisa_link_simulation.py` and `scripts/tdi_wdm_pipeline.py` remain compatibility wrappers for the full pipeline entry points.
- `examples/example_lisa_link_simulation.ipynb` shows the main link/TDI pipeline.
- `examples/example_tdi_wdm_pipeline.ipynb` shows the TDI/WDM and T-weighted stage.
- `examples/smoke_test_generated_waveform.py` runs a fast generated-waveform smoke test without regenerating the full noisy products.

Example:

```python
from scripts.tdi_wdm_pipeline import TDIWDMPipelineConfig, run_tdi_wdm_pipeline

products = run_tdi_wdm_pipeline(
    "lisa_link_sim/script_run",
    config=TDIWDMPipelineConfig(make_grid_plots=True, make_t_weighted_plots=True),
)
products.wdm_summary.sort_values("signal_over_offsource_mean", ascending=False).head()
```

Fast smoke check:

```bash
conda run -n lisa-tf python examples/smoke_test_generated_waveform.py
```

## Key Dependencies

- **Python 3.12**
- **lisaorbits** - LISA orbital mechanics
- **lisagwresponse** - Gravitational wave response functions
- **lisainstrument** - LISA instrument modeling
- **pytdi** - Time-delay interferometry
- **Scientific stack**: NumPy, SciPy, Pandas, Matplotlib, HDF5

## Development

To add new dependencies:
1. Install with conda or pip
2. Update `env.yml` with the new package
3. Commit and push the updated `env.yml`

## Output Files

Generated files are stored in `lisa_link_sim/outputs/`:
- `*.csv` - CSV data files
- `*.h5` - HDF5 data files
- `plots/` - Visualization outputs
- `diagnostics/` - Diagnostic information

## Notes

- Large data files (*.h5, *.npz, *.csv) are excluded from git via `.gitignore`
- Store large datasets separately or use Git LFS if needed
- See individual notebook documentation for detailed analysis steps

## License

[Add your license here]

## Contact

[Add contact information if needed]
