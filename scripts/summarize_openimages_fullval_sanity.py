from __future__ import annotations

from pathlib import Path

from summarize_openimages_fullfiltered_sanity import main as _main


LEGACY_FULLVAL_GLOB = "outputs/openimages_fullval_filtered_heldout_ultrastrict*/calibrated_baseline_summary.csv"
CANONICAL_FULLFILTERED_GLOB = "outputs/openimages_fullfiltered_validation_heldout_ultrastrict*/calibrated_baseline_summary.csv"


def choose_default_glob() -> str:
    """
    Keep the historic `fullval` wrapper name working while preferring the
    canonical filtered-validation output prefix when it exists.
    """
    canonical_matches = list(Path().glob(CANONICAL_FULLFILTERED_GLOB))
    if canonical_matches:
        return CANONICAL_FULLFILTERED_GLOB
    return LEGACY_FULLVAL_GLOB


def main() -> int:
    return _main(
        default_glob=choose_default_glob(),
        default_output_dir="results/supplementary",
        default_dataset_name="Larger filtered Open Images validation-subset check",
        default_dataset_id="openimages_full_filtered_validation",
        default_per_config_name="openimages_full_filtered_validation_sanity_per_config.csv",
        default_summary_name="openimages_full_filtered_validation_sanity_summary.csv",
        default_report_name="openimages_full_filtered_validation_sanity_report.md",
    )


if __name__ == "__main__":
    raise SystemExit(main())
