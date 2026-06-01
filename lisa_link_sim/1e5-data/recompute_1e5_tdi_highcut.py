from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import (  # noqa: E402
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    compute_tdi_from_gws,
    compute_tdi_from_instrument,
    plot_noisy_with_gw_signal,
)


def main() -> None:
    work_dir = PROJECT_ROOT / "lisa_link_sim" / "1e5-data"
    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)

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

    noisy_diag = compute_tdi_from_instrument(config, paths, make_plots=True)
    gw_diag = compute_tdi_from_gws(config, paths, make_plots=True)
    overlay_diag = plot_noisy_with_gw_signal(paths, suffix=str(config.tdi_generation))

    diagnostics = {
        "frequency_low_hz": config.frequency_low_hz,
        "frequency_high_hz": config.frequency_high_hz,
        **noisy_diag,
        **gw_diag,
        **overlay_diag,
    }
    diagnostics_file = paths.output_dir / "tdi_highcut_1e-1_diagnostics.json"
    diagnostics_file.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))
    print(f"saved {diagnostics_file}")


if __name__ == "__main__":
    main()
