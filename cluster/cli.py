"""Command line interface for SMBHB WDM clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

from .autoencoder import train_autoencoder
from .clustering import run_hdbscan
from .cleanup import cleanup_work_dirs, discover_completed_work_dirs, format_bytes, summarize_cleanup
from .config import load_pipeline_config
from .evaluation import evaluate_clusters
from .parameters import read_catalog, sample_smbhb_catalog, write_catalog
from .psd import estimate_noise_psd_from_work_dir, load_noise_psd
from .simulation import run_catalog_simulations
from .wdm_pixels import build_dataset, discover_work_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SMBHB A/E WDM autoencoder + HDBSCAN pipeline.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="Sample an SMBHB parameter catalog.")
    sample.add_argument("--n-sources", type=int, required=True)
    sample.add_argument("--seed", type=int, default=12345)
    sample.add_argument("--out", type=Path, default=None)

    simulate = sub.add_parser("simulate", help="Generate waveforms and run LISA simulations for a catalog.")
    simulate.add_argument("--catalog", type=Path, default=None)
    simulate.add_argument("--runs-root", type=Path, default=None)
    simulate.add_argument("--start-index", type=int, default=0, help="First catalog row to simulate, zero-based.")
    simulate.add_argument("--limit", type=int, default=None)
    simulate.add_argument("--force-waveform", action="store_true")
    simulate.add_argument("--force-simulation", action="store_true")
    simulate.add_argument("--make-plots", action="store_true")

    psd = sub.add_parser("estimate-psd", help="Estimate off-source A/E PSD from one simulation.")
    psd.add_argument("--reference-work-dir", type=Path, default=None)
    psd.add_argument("--out", type=Path, default=None)

    dataset = sub.add_parser("build-dataset", help="Build fixed-size A/E WDM patch dataset.")
    dataset.add_argument("--runs-root", type=Path, default=None)
    dataset.add_argument("--work-dir", type=Path, action="append", default=None)
    dataset.add_argument("--catalog", type=Path, default=None)
    dataset.add_argument("--psd-csv", type=Path, default=None)
    dataset.add_argument("--out", type=Path, default=None)
    dataset.add_argument("--metadata-out", type=Path, default=None)
    dataset.add_argument(
        "--cleanup-after-build",
        action="store_true",
        help="Delete large per-source simulation intermediates after each patch is built.",
    )

    train = sub.add_parser("train-autoencoder", help="Train CNN autoencoder and save embeddings.")
    train.add_argument("--dataset", type=Path, default=None)

    hdb = sub.add_parser("cluster", help="Run HDBSCAN on embeddings.")
    hdb.add_argument("--embeddings", type=Path, default=None)

    evaluate = sub.add_parser("evaluate", help="Evaluate cluster labels against metadata.")
    evaluate.add_argument("--embeddings", type=Path, default=None)
    evaluate.add_argument("--labels", type=Path, default=None)
    evaluate.add_argument("--metadata", type=Path, default=None)

    smoke = sub.add_parser("smoke", help="Run lightweight smoke checks.")
    smoke.add_argument("--with-ml", action="store_true")

    cleanup = sub.add_parser("cleanup", help="Delete large simulation intermediates after patches are built.")
    cleanup.add_argument("--runs-root", type=Path, default=None)
    cleanup.add_argument("--work-dir", type=Path, action="append", default=None)
    cleanup.add_argument("--apply", action="store_true", help="Actually delete files. Without this, only prints a dry run.")

    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)

    if args.command == "sample":
        catalog = sample_smbhb_catalog(args.n_sources, config.parameters, seed=args.seed)
        path = write_catalog(catalog, args.out or config.dataset.catalog_csv)
        print(path)
    elif args.command == "simulate":
        catalog = read_catalog(args.catalog or config.dataset.catalog_csv)
        work_dirs = run_catalog_simulations(
            catalog,
            runs_root=args.runs_root or config.dataset.runs_root,
            waveform_config=config.waveform,
            simulation_config=config.simulation,
            start_index=args.start_index,
            limit=args.limit,
            force_waveform=args.force_waveform,
            force_simulation=args.force_simulation,
            make_plots=args.make_plots,
        )
        for path in work_dirs:
            print(path)
    elif args.command == "estimate-psd":
        out = args.out or config.psd.output_csv
        psd_df = estimate_noise_psd_from_work_dir(
            args.reference_work_dir or config.psd.reference_work_dir,
            config=config.psd,
            tdi_generation=config.simulation.tdi_generation,
            output_csv=out,
        )
        print(f"{out} ({len(psd_df)} frequency bins)")
    elif args.command == "build-dataset":
        if args.work_dir:
            work_dirs = args.work_dir
        else:
            work_dirs = discover_work_dirs(args.runs_root or config.dataset.runs_root)
        catalog = read_catalog(args.catalog or config.dataset.catalog_csv) if (args.catalog or Path(config.dataset.catalog_csv).exists()) else None
        psd_df = load_noise_psd(args.psd_csv or config.psd.output_csv)
        out_npz, out_meta = build_dataset(
            work_dirs,
            psd=psd_df,
            wdm_config=config.wdm,
            psd_config=config.psd,
            output_npz=args.out or config.dataset.dataset_npz,
            metadata_csv=args.metadata_out or config.dataset.metadata_csv,
            catalog=catalog,
            tdi_generation=config.simulation.tdi_generation,
            cleanup_after_build=args.cleanup_after_build,
        )
        print(out_npz)
        print(out_meta)
    elif args.command == "train-autoencoder":
        model, embeddings, history = train_autoencoder(
            args.dataset or config.dataset.dataset_npz,
            latent_dim=config.autoencoder.latent_dim,
            epochs=config.autoencoder.epochs,
            batch_size=config.autoencoder.batch_size,
            learning_rate=config.autoencoder.learning_rate,
            weight_decay=config.autoencoder.weight_decay,
            validation_fraction=config.autoencoder.validation_fraction,
            seed=config.autoencoder.seed,
            model_path=config.autoencoder.model_path,
            embeddings_npz=config.autoencoder.embeddings_npz,
            history_csv=config.autoencoder.history_csv,
        )
        print(model)
        print(embeddings)
        print(history)
    elif args.command == "cluster":
        labels = run_hdbscan(
            args.embeddings or config.autoencoder.embeddings_npz,
            min_cluster_size=config.clustering.min_cluster_size,
            min_samples=config.clustering.min_samples,
            metric=config.clustering.metric,
            labels_csv=config.clustering.labels_csv,
        )
        print(labels)
    elif args.command == "evaluate":
        metrics, summary = evaluate_clusters(
            embeddings_npz=args.embeddings or config.autoencoder.embeddings_npz,
            labels_csv=args.labels or config.clustering.labels_csv,
            metadata_csv=args.metadata or config.dataset.metadata_csv,
            metrics_json=config.clustering.evaluation_json,
            summary_csv=config.clustering.summary_csv,
        )
        print(metrics)
        print(summary)
    elif args.command == "smoke":
        from .smoke_checks import run_smoke_checks

        run_smoke_checks(with_ml=args.with_ml)
    elif args.command == "cleanup":
        if args.work_dir:
            work_dirs = args.work_dir
        else:
            work_dirs = discover_completed_work_dirs(args.runs_root or config.dataset.runs_root)
        results = cleanup_work_dirs(work_dirs, apply=args.apply, require_diagnostics=True)
        summary = summarize_cleanup(results)
        action = "deleted" if args.apply else "would_delete"
        print(
            f"{action}: {summary['files']} files, {format_bytes(summary['bytes'])}, "
            f"{summary['work_dirs']} work dirs, {summary['skipped']} skipped"
        )
        for result in results:
            if result.skipped:
                print(f"skip {result.work_dir}: {result.reason}")
            else:
                print(f"{result.work_dir}: {len(result.deleted_files)} files, {format_bytes(result.deleted_bytes)}")


if __name__ == "__main__":
    main()
