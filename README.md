# LISA-TF: Gravitational Wave Detection Analysis

Analysis and simulation code for LISA (Laser Interferometer Space Antenna) gravitational wave detection using time-delay interferometry (TDI).

## Project Structure

```
├── lisa_link_sim/              # Main simulation notebooks and data
│   ├── *.ipynb                # Jupyter notebooks for analysis
│   ├── inputs/                # Input data files
│   └── outputs/               # Generated output files
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
