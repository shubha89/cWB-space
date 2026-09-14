"""Reproduce the nine numerical figures and copy the supplied Figure 1 artwork."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT or out.is_relative_to(ROOT/'data') or out.is_relative_to(ROOT/'assets'):
        parser.error('Choose an output directory separate from source inputs.')
    out.mkdir(parents=True, exist_ok=True)
    for script in ['plot_strength.py', 'plot_response.py', 'plot_reconstruction.py', 'plot_sangria.py']:
        subprocess.run([sys.executable, str(ROOT/'examples'/script), '--output-dir', str(out)], check=True)
    shutil.copy2(ROOT/'assets/figure01_pipeline.pdf', out/'figure01_pipeline.pdf')
    print(f'Completed figure outputs in {out}')

if __name__ == '__main__':
    main()
