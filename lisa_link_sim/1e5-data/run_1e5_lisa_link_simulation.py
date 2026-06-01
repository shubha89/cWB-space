from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import (  # noqa: E402
    LisaLinkSimulationConfig,
    load_npz_waveform,
    run_lisa_link_simulation,
)


def main() -> None:
    waveform_file = PROJECT_ROOT / "lisa_link_sim" / "1e5-wave" / "inputs" / "my_waveform_1e5.npz"
    work_dir = PROJECT_ROOT / "lisa_link_sim" / "1e5-data"

    t, hplus, hcross = load_npz_waveform(waveform_file)

    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        ra=3.141592653589793 / 4.0,
        dec=0.3,
        central_freq_hz=2.816e14,
        orbit_name="esa-trailing",
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=12345,
        frequency_low_hz=1e-5,
        frequency_high_hz=1e-1,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
    )

    products = run_lisa_link_simulation(
        t,
        hplus,
        hcross,
        work_dir=work_dir,
        config=config,
        make_plots=False,
    )

    diagnostics_file = work_dir / "outputs" / "simulation_diagnostics.json"
    diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_file.write_text(json.dumps(dict(products.diagnostics), indent=2))

    print(f"waveform_file = {waveform_file}")
    print(f"work_dir      = {work_dir}")
    print(f"diagnostics   = {diagnostics_file}")
    print(json.dumps(dict(products.diagnostics), indent=2))


if __name__ == "__main__":
    main()
