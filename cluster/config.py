"""Configuration objects for the SMBHB WDM clustering workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def coerce_eta(value: float) -> float:
    """Accept repo label notation: 1.5 -> 0.15 and 2.5 -> 0.25."""

    value = float(value)
    if value > 1.0:
        value = value / 10.0
    if not (0.0 < value <= 0.25):
        raise ValueError("Symmetric mass ratio eta must be in (0, 0.25].")
    return value


@dataclass(frozen=True)
class SMBHBParameterRange:
    """Sampling range for SMBHB waveform parameters."""

    total_mass_min: float = 1.0e4
    total_mass_max: float = 1.0e8
    eta_min: float = 0.15
    eta_max: float = 0.25
    distance_mpc_min: float = 1.0e3
    distance_mpc_max: float = 2.0e4
    spin1_min: float = -0.5
    spin1_max: float = 0.5
    spin2_min: float = -0.5
    spin2_max: float = 0.5
    iota_min: float = 0.0
    iota_max: float = 3.141592653589793
    ra_min: float = 0.0
    ra_max: float = 6.283185307179586
    dec_min: float = -1.5707963267948966
    dec_max: float = 1.5707963267948966
    phi_ref_min: float = 0.0
    phi_ref_max: float = 6.283185307179586
    f_min_hz: float = 1.0e-4

    @property
    def eta_bounds(self) -> tuple[float, float]:
        eta_min = coerce_eta(self.eta_min)
        eta_max = coerce_eta(self.eta_max)
        if eta_min > eta_max:
            raise ValueError("eta_min must be <= eta_max.")
        return eta_min, eta_max

    def validate(self) -> None:
        if self.total_mass_min <= 0.0 or self.total_mass_max <= 0.0:
            raise ValueError("Total mass bounds must be positive.")
        if self.total_mass_min > self.total_mass_max:
            raise ValueError("total_mass_min must be <= total_mass_max.")
        _ = self.eta_bounds


@dataclass(frozen=True)
class WaveformConfig:
    raw_dt_s: float = 1.0
    raw_half_duration_s: float = 300_000.0
    default_model: str = "IMRPhenomTHM"


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float = 0.1
    science_duration_s: float = 150_000.0
    burn_in_seconds: float = 600.0
    orbit_dt_s: float = 600.0
    orbit_duration_factor: float = 5.0
    central_freq_hz: float = 2.816e14
    frequency_low_hz: float = 1.0e-5
    frequency_high_hz: float = 1.0e-1
    tdi_generation: int = 2
    interpolation_order: int = 31
    seed: int = 12345


@dataclass(frozen=True)
class PSDConfig:
    reference_work_dir: str = "lisa_link_sim/eta-1_5"
    offsource_guard_seconds: float = 500.0
    nperseg: int = 16_384
    min_psd: float = 1.0e-300
    output_csv: str = "cluster/runs/noise_psd.csv"


@dataclass(frozen=True)
class WDMMapConfig:
    target_dt_s: float = 1.0
    M: int = 512
    K: int = 32
    beta_order: int = 6
    precision: int = 6
    backend: str = "numba"
    freq_min_hz: float = 1.0e-5
    freq_max_hz: float = 1.0e-1
    patch_freq_bins: int = 128
    patch_time_bins: int = 128
    patch_duration_seconds: float = 30_000.0
    log_scale: bool = True
    robust_clip: float = 8.0


@dataclass(frozen=True)
class DatasetConfig:
    runs_root: str = "cluster/runs/simulations"
    catalog_csv: str = "cluster/runs/catalog.csv"
    dataset_npz: str = "cluster/runs/wdm_patches.npz"
    metadata_csv: str = "cluster/runs/wdm_patches_metadata.csv"


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_dim: int = 64
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    validation_fraction: float = 0.15
    seed: int = 12345
    model_path: str = "cluster/runs/autoencoder.pt"
    embeddings_npz: str = "cluster/runs/embeddings.npz"
    history_csv: str = "cluster/runs/training_history.csv"


@dataclass(frozen=True)
class ClusteringConfig:
    min_cluster_size: int = 5
    min_samples: int | None = None
    metric: str = "euclidean"
    labels_csv: str = "cluster/runs/hdbscan_labels.csv"
    evaluation_json: str = "cluster/runs/cluster_metrics.json"
    summary_csv: str = "cluster/runs/cluster_summary.csv"


@dataclass(frozen=True)
class PipelineConfig:
    parameters: SMBHBParameterRange = field(default_factory=SMBHBParameterRange)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    psd: PSDConfig = field(default_factory=PSDConfig)
    wdm: WDMMapConfig = field(default_factory=WDMMapConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pipeline_config(path: str | Path | None) -> PipelineConfig:
    """Load a YAML config, falling back to defaults when no path is given."""

    if path is None:
        return PipelineConfig()
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError("Loading YAML configs requires pyyaml.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    def section(cls: type, name: str):
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise TypeError(f"Config section {name!r} must be a mapping.")
        return cls(**values)

    return PipelineConfig(
        parameters=section(SMBHBParameterRange, "parameters"),
        waveform=section(WaveformConfig, "waveform"),
        simulation=section(SimulationConfig, "simulation"),
        psd=section(PSDConfig, "psd"),
        wdm=section(WDMMapConfig, "wdm"),
        dataset=section(DatasetConfig, "dataset"),
        autoencoder=section(AutoencoderConfig, "autoencoder"),
        clustering=section(ClusteringConfig, "clustering"),
    )
