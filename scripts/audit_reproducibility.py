from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CsvExpectation:
    path: str
    columns: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    numeric: tuple["NumericExpectation", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NumericExpectation:
    match: tuple[tuple[str, str], ...]
    values: tuple[tuple[str, float], ...]
    tolerance: float = 5e-4


@dataclass(frozen=True)
class ClaimCheck:
    name: str
    scripts: tuple[str, ...] = ()
    configs: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    csv: tuple[CsvExpectation, ...] = field(default_factory=tuple)
    note: str = ""


CHECKS: tuple[ClaimCheck, ...] = (
    ClaimCheck(
        name="Open Images 10k main held-out protocol",
        scripts=(
            "scripts/run_openimages_pilot.py",
            "scripts/run_openimages_heldout_calibration.py",
            "scripts/run_openimages_calibrated_baselines.py",
            "scripts/analyze_openimages_heldout_groups.py",
            "scripts/summarize_training_and_main_results.py",
        ),
        configs=(
            "configs/openimages_10k_heldout_ultrastrict.yaml",
            "configs/openimages_10k_heldout_ultrastrict_classB.yaml",
        ),
        csv=(
            CsvExpectation(
                "results/openimages_10k/main_method_12config_summary.csv",
                methods=(
                    "clip_knn_global_threshold",
                    "clip_knn_class_thresholds",
                    "clip_knn_isotonic_global_threshold",
                    "heldout_gate_global_threshold",
                    "heldout_gate_class_thresholds",
                ),
                numeric=(
                    NumericExpectation((("method", "clip_knn_global_threshold"),), (("average_precision_mean", 0.8519), ("best_f1_mean", 0.7713), ("tecr_mean", 0.2592), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "clip_knn_class_thresholds"),), (("tecr_mean", 0.2391), ("tecr_reduction_pct", 7.8)), 5e-4),
                    NumericExpectation((("method", "clip_knn_isotonic_global_threshold"),), (("tecr_mean", 0.2352), ("tecr_reduction_pct", 9.3)), 5e-4),
                    NumericExpectation((("method", "heldout_gate_global_threshold"),), (("average_precision_mean", 0.8520), ("best_f1_mean", 0.7704), ("tecr_mean", 0.2126), ("tecr_reduction_pct", 18.0)), 5e-4),
                    NumericExpectation((("method", "heldout_gate_class_thresholds"),), (("tecr_mean", 0.2206), ("tecr_reduction_pct", 14.9)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/openimages_10k/training_baseline_12config_summary.csv",
                methods=("asl_class_thresholds", "db_loss_class_thresholds"),
                numeric=(
                    NumericExpectation((("method", "asl_class_thresholds"),), (("average_precision_mean", 0.8421), ("best_f1_mean", 0.7734), ("tecr_mean", 0.2490)), 5e-4),
                    NumericExpectation((("method", "db_loss_class_thresholds"),), (("average_precision_mean", 0.8363), ("best_f1_mean", 0.7674), ("tecr_mean", 0.2573)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="COCO val2017 main held-out protocol",
        scripts=(
            "scripts/run_coco_pilot.py",
            "scripts/run_coco_heldout_calibration.py",
            "scripts/run_coco_training_baselines.py",
            "scripts/summarize_training_and_main_results.py",
        ),
        configs=(
            "configs/coco_heldout_ultrastrict.yaml",
            "configs/coco_heldout_ultrastrict_classB.yaml",
        ),
        csv=(
            CsvExpectation(
                "results/coco_val2017/main_method_12config_summary.csv",
                methods=(
                    "clip_knn_global_threshold",
                    "clip_knn_class_thresholds",
                    "clip_knn_isotonic_global_threshold",
                    "heldout_gate_global_threshold",
                    "heldout_gate_class_thresholds",
                ),
                numeric=(
                    NumericExpectation((("method", "clip_knn_global_threshold"),), (("average_precision_mean", 0.8756), ("best_f1_mean", 0.7887), ("tecr_mean", 0.2545), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "clip_knn_class_thresholds"),), (("tecr_mean", 0.2368), ("tecr_reduction_pct", 7.0)), 5e-4),
                    NumericExpectation((("method", "clip_knn_isotonic_global_threshold"),), (("tecr_mean", 0.2258), ("tecr_reduction_pct", 11.3)), 5e-4),
                    NumericExpectation((("method", "heldout_gate_global_threshold"),), (("average_precision_mean", 0.8805), ("best_f1_mean", 0.7934), ("tecr_mean", 0.1848), ("tecr_reduction_pct", 27.4)), 5e-4),
                    NumericExpectation((("method", "heldout_gate_class_thresholds"),), (("tecr_mean", 0.1840), ("tecr_reduction_pct", 27.7)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/coco_val2017/training_baseline_12config_summary.csv",
                methods=("asl_class_thresholds", "db_loss_class_thresholds"),
                numeric=(
                    NumericExpectation((("method", "text_init_asl_class_thresholds"),), (("average_precision_mean", 0.8828), ("best_f1_mean", 0.7987), ("tecr_mean", 0.2213)), 5e-4),
                    NumericExpectation((("method", "asl_class_thresholds"),), (("average_precision_mean", 0.8846), ("best_f1_mean", 0.8002), ("tecr_mean", 0.2220)), 5e-4),
                    NumericExpectation((("method", "db_loss_class_thresholds"),), (("average_precision_mean", 0.8788), ("best_f1_mean", 0.7952), ("tecr_mean", 0.2247)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="Open Images gate ablation",
        scripts=("scripts/run_openimages_gate_ablation.py",),
        configs=(
            "configs/openimages_10k_heldout_ultrastrict.yaml",
            "configs/openimages_10k_heldout_ultrastrict_classB.yaml",
        ),
        csv=(
            CsvExpectation(
                "results/openimages_10k/gate_ablation_12config_summary.csv",
                methods=(
                    "clip_knn",
                    "pure_residual_heldout",
                    "full_gate_tol_0.00pct",
                    "full_gate_tol_0.25pct",
                    "full_gate_tol_0.50pct",
                    "full_gate_tol_1.00pct",
                ),
                numeric=(
                    NumericExpectation((("method", "clip_knn"),), (("average_precision_mean", 0.8519), ("best_f1_mean", 0.7713), ("tecr_mean", 0.2592), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "pure_residual_heldout"),), (("tecr_mean", 0.2464), ("tecr_reduction_pct", 5.1)), 5e-4),
                    NumericExpectation((("method", "full_gate_tol_0.00pct"),), (("tecr_mean", 0.2233), ("tecr_reduction_pct", 14.0)), 5e-4),
                    NumericExpectation((("method", "full_gate_tol_0.25pct"),), (("tecr_mean", 0.2126), ("tecr_reduction_pct", 18.0)), 5e-4),
                    NumericExpectation((("method", "full_gate_tol_0.50pct"),), (("tecr_mean", 0.2081), ("tecr_reduction_pct", 19.7)), 5e-4),
                    NumericExpectation((("method", "full_gate_tol_1.00pct"),), (("tecr_mean", 0.2036), ("tecr_reduction_pct", 21.5)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="Open Images calibration-size sensitivity",
        scripts=(
            "scripts/run_openimages_calibration_size_sensitivity.py",
            "scripts/summarize_calibration_size_sensitivity.py",
        ),
        configs=(
            "configs/openimages_10k_heldout_ultrastrict.yaml",
            "configs/openimages_10k_heldout_ultrastrict_classB.yaml",
        ),
        csv=(
            CsvExpectation(
                "results/openimages_10k/calibration_size_12config_summary.csv",
                columns=("calibration_fraction_used", "method", "tecr_mean"),
                methods=("clip_knn", "heldout_gate"),
                numeric=(
                    NumericExpectation((("calibration_fraction_used", "0.125"), ("method", "heldout_gate")), (("average_precision_mean", 0.8519), ("best_f1_mean", 0.7662), ("tecr_mean", 0.2331), ("tecr_reduction_pct_vs_main_clip_knn", 10.1)), 5e-4),
                    NumericExpectation((("calibration_fraction_used", "0.25"), ("method", "heldout_gate")), (("best_f1_mean", 0.7685), ("tecr_mean", 0.2238), ("tecr_reduction_pct_vs_main_clip_knn", 13.7)), 5e-4),
                    NumericExpectation((("calibration_fraction_used", "0.5"), ("method", "heldout_gate")), (("best_f1_mean", 0.7698), ("tecr_mean", 0.2185), ("tecr_reduction_pct_vs_main_clip_knn", 15.7)), 5e-4),
                    NumericExpectation((("calibration_fraction_used", "1.0"), ("method", "heldout_gate")), (("best_f1_mean", 0.7704), ("tecr_mean", 0.2126), ("tecr_reduction_pct_vs_main_clip_knn", 18.0)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="Open Images calibration-ratio sensitivity",
        scripts=(
            "scripts/run_openimages_calibration_ratio_sensitivity.py",
            "scripts/summarize_calibration_ratio_sensitivity.py",
        ),
        configs=("configs/openimages_10k_heldout_ultrastrict.yaml",),
        csv=(
            CsvExpectation(
                "results/supplementary/calibration_ratio_summary.csv",
                columns=("calibration_dataset_fraction", "method", "num_configs", "num_split_averages", "tecr_mean"),
                methods=("clip_knn", "heldout_gate"),
                numeric=(
                    NumericExpectation((("calibration_dataset_fraction", "0.1"), ("method", "clip_knn")), (("num_configs", 12), ("num_split_averages", 60), ("average_precision_mean", 0.8525), ("best_f1_mean", 0.7713), ("tecr_mean", 0.2584), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.1"), ("method", "heldout_gate")), (("num_configs", 12), ("num_split_averages", 60), ("average_precision_mean", 0.8528), ("best_f1_mean", 0.7701), ("tecr_mean", 0.2130), ("tecr_reduction_pct", 17.6)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.15"), ("method", "clip_knn")), (("average_precision_mean", 0.8525), ("best_f1_mean", 0.7707), ("tecr_mean", 0.2529), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.15"), ("method", "heldout_gate")), (("average_precision_mean", 0.8527), ("best_f1_mean", 0.7702), ("tecr_mean", 0.2077), ("tecr_reduction_pct", 17.9)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.2"), ("method", "clip_knn")), (("average_precision_mean", 0.8525), ("best_f1_mean", 0.7721), ("tecr_mean", 0.2568), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.2"), ("method", "heldout_gate")), (("average_precision_mean", 0.8527), ("best_f1_mean", 0.7711), ("tecr_mean", 0.2168), ("tecr_reduction_pct", 15.6)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.25"), ("method", "clip_knn")), (("average_precision_mean", 0.8525), ("best_f1_mean", 0.7719), ("tecr_mean", 0.2512), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.25"), ("method", "heldout_gate")), (("average_precision_mean", 0.8528), ("best_f1_mean", 0.7707), ("tecr_mean", 0.2078), ("tecr_reduction_pct", 17.3)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.3"), ("method", "clip_knn")), (("average_precision_mean", 0.8525), ("best_f1_mean", 0.7721), ("tecr_mean", 0.2507), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("calibration_dataset_fraction", "0.3"), ("method", "heldout_gate")), (("average_precision_mean", 0.8527), ("best_f1_mean", 0.7705), ("tecr_mean", 0.2047), ("tecr_reduction_pct", 18.4)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="ASL+gate add-on",
        scripts=(
            "scripts/run_asl_gate_baselines.py",
            "scripts/summarize_asl_gate_results.py",
        ),
        configs=(
            "configs/openimages_10k_heldout_ultrastrict.yaml",
            "configs/coco_heldout_ultrastrict.yaml",
            "configs/nuswide_heldout_ultrastrict.yaml",
        ),
        csv=(
            CsvExpectation(
                "results/openimages_10k/asl_gate_summary.csv",
                methods=("asl_global_threshold", "asl_gate_global_threshold"),
                numeric=(
                    NumericExpectation((("method", "asl_global_threshold"),), (("average_precision_mean", 0.8421), ("best_f1_mean", 0.7729), ("tecr_mean", 0.2629), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "asl_gate_global_threshold"),), (("average_precision_mean", 0.8419), ("best_f1_mean", 0.7722), ("tecr_mean", 0.2249), ("tecr_reduction_pct", 14.5)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/coco_val2017/asl_gate_summary.csv",
                methods=("asl_global_threshold", "asl_gate_global_threshold"),
                numeric=(
                    NumericExpectation((("method", "asl_global_threshold"),), (("average_precision_mean", 0.8846), ("best_f1_mean", 0.7988), ("tecr_mean", 0.2508), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "asl_gate_global_threshold"),), (("average_precision_mean", 0.8895), ("best_f1_mean", 0.8004), ("tecr_mean", 0.1912), ("tecr_reduction_pct", 23.7)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/nuswide/asl_gate_10config_summary.csv",
                methods=("asl_global_threshold", "asl_gate_global_threshold"),
                numeric=(
                    NumericExpectation((("method", "asl_global_threshold"),), (("average_precision_mean", 0.8004), ("best_f1_mean", 0.7474), ("tecr_mean", 0.2518), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "asl_gate_global_threshold"),), (("average_precision_mean", 0.7989), ("best_f1_mean", 0.7469), ("tecr_mean", 0.2249), ("tecr_reduction_pct", 10.7)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="NUS-WIDE stress test and supplementary suite",
        scripts=("scripts/run_nuswide_full_suite.py", "scripts/summarize_nuswide_full_suite.py", "scripts/nuswide_common.py"),
        configs=(
            "configs/nuswide_heldout_ultrastrict.yaml",
            "configs/nuswide_heldout_ultrastrict_classB.yaml",
        ),
        results=(
            "data_manifest/nuswide_image_names.json",
            "data_manifest/nuswide_classes.json",
        ),
        csv=(
            CsvExpectation(
                "results/nuswide/main_summary_10config.csv",
                methods=("clip_knn_global_threshold", "heldout_gate_global_threshold"),
                numeric=(
                    NumericExpectation((("method", "clip_knn_global_threshold"),), (("ap_mean", 0.7863), ("f1_mean", 0.7339), ("tecr_mean", 0.2571), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "clip_knn_class_thresholds"),), (("tecr_mean", 0.2363), ("tecr_reduction_pct", 8.1)), 5e-4),
                    NumericExpectation((("method", "clip_knn_isotonic_global_threshold"),), (("tecr_mean", 0.2427), ("tecr_reduction_pct", 5.6)), 5e-4),
                    NumericExpectation((("method", "heldout_gate_global_threshold"),), (("ap_mean", 0.7861), ("f1_mean", 0.7310), ("tecr_mean", 0.1920), ("tecr_reduction_pct", 25.3)), 5e-4),
                    NumericExpectation((("method", "heldout_gate_class_thresholds"),), (("tecr_mean", 0.1995), ("tecr_reduction_pct", 22.4)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/nuswide/training_baseline_10config_summary.csv",
                methods=("asl_class_thresholds", "db_loss_class_thresholds"),
                numeric=(
                    NumericExpectation((("method", "asl_class_thresholds"),), (("average_precision_mean", 0.8004), ("best_f1_mean", 0.7503), ("tecr_mean", 0.2157)), 5e-4),
                    NumericExpectation((("method", "db_loss_class_thresholds"),), (("average_precision_mean", 0.7935), ("best_f1_mean", 0.7396), ("tecr_mean", 0.2319)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/nuswide/frequency_group_10config_summary.csv",
                columns=("frequency_group", "clip_knn_tecr_mean", "heldout_tecr_mean", "tecr_reduction_pct", "tecr_reduction_pct_mean_across_configs"),
                numeric=(
                    NumericExpectation((("frequency_group", "head"),), (("num_emerging_labels_mean", 5.1), ("clip_knn_tecr_mean", 0.2011), ("heldout_tecr_mean", 0.1629), ("tecr_delta", -0.0382), ("tecr_reduction_pct_mean_across_configs", 18.2)), 5e-4),
                    NumericExpectation((("frequency_group", "mid"),), (("num_emerging_labels_mean", 5.2), ("clip_knn_tecr_mean", 0.0177), ("heldout_tecr_mean", 0.0156), ("tecr_delta", -0.0021), ("tecr_reduction_pct_mean_across_configs", 5.2)), 5e-4),
                    NumericExpectation((("frequency_group", "tail"),), (("num_emerging_labels_mean", 4.7), ("clip_knn_tecr_mean", 0.0137), ("heldout_tecr_mean", 0.0105), ("tecr_delta", -0.0032), ("tecr_reduction_pct_mean_across_configs", 26.3)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/nuswide/gate_ablation_10config_summary.csv",
                methods=("clip_knn", "residual_only_gate", "full_gate"),
            ),
        ),
    ),
    ClaimCheck(
        name="TECR definition robustness",
        scripts=("scripts/run_tecr_robustness.py",),
        configs=("configs/openimages_10k_heldout_ultrastrict.yaml",),
        csv=(
            CsvExpectation(
                "results/supplementary/tecr_robustness_summary.csv",
                columns=("variant", "method", "tecr_mean"),
                methods=("clip_knn", "heldout_gate"),
                numeric=(
                    NumericExpectation((("variant", "tail10_emerging15"), ("method", "heldout_gate")), (("tecr_mean", 0.1971), ("tecr_reduction_pct", 28.1)), 5e-4),
                    NumericExpectation((("variant", "tail20_emerging10"), ("method", "heldout_gate")), (("tecr_mean", 0.1433), ("tecr_reduction_pct", 18.7)), 5e-4),
                    NumericExpectation((("variant", "tail20_emerging15"), ("method", "heldout_gate")), (("tecr_mean", 0.1863), ("tecr_reduction_pct", 28.4)), 5e-4),
                    NumericExpectation((("variant", "tail20_emerging20"), ("method", "heldout_gate")), (("tecr_mean", 0.2953), ("tecr_reduction_pct", 25.5)), 5e-4),
                    NumericExpectation((("variant", "tail30_emerging15"), ("method", "heldout_gate")), (("tecr_mean", 0.1796), ("tecr_reduction_pct", 29.2)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="Alternative post-hoc and selective baselines",
        scripts=(
            "scripts/run_openimages_posthoc_baselines.py",
            "scripts/run_openimages_selective_baseline.py",
            "scripts/run_openimages_odin_baseline.py",
        ),
        configs=("configs/openimages_10k_heldout_ultrastrict.yaml",),
        csv=(
            CsvExpectation(
                "results/supplementary/posthoc_combined_summary.csv",
                methods=(
                    "clip_knn",
                    "temperature_scaling",
                    "split_conformal",
                    "maxlogit_known_reject",
                    "entropy_reject",
                    "elta_confidence_heldout",
                ),
                numeric=(
                    NumericExpectation((("method", "clip_knn"),), (("average_precision_mean", 0.8519), ("best_f1_mean", 0.7713), ("tecr_mean", 0.2592), ("avg_predicted_labels_mean", 0.7561), ("coverage_mean", 0.8329), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "temperature_scaling"),), (("tecr_mean", 0.2592), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "split_conformal"),), (("best_f1_mean", 0.2274), ("tecr_mean", 0.0014), ("avg_predicted_labels_mean", 0.0618), ("coverage_mean", 0.1312), ("tecr_reduction_pct", 99.5)), 5e-4),
                    NumericExpectation((("method", "maxlogit_known_reject"),), (("tecr_mean", 0.2592), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "entropy_reject"),), (("average_precision_mean", 0.8513), ("best_f1_mean", 0.7710), ("tecr_mean", 0.2562), ("tecr_reduction_pct", 1.1)), 5e-4),
                    NumericExpectation((("method", "elta_confidence_heldout"),), (("average_precision_mean", 0.8520), ("best_f1_mean", 0.7704), ("tecr_mean", 0.2126), ("avg_predicted_labels_mean", 0.6970), ("coverage_mean", 0.8021), ("tecr_reduction_pct", 18.0)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/supplementary/odin_combined_summary.csv",
                methods=("clip_knn", "odin_low_reject"),
                numeric=(
                    NumericExpectation((("method", "odin_low_reject"),), (("average_precision_mean", 0.8517), ("best_f1_mean", 0.7712), ("tecr_mean", 0.2592), ("avg_predicted_labels_mean", 0.7558), ("coverage_mean", 0.8326), ("tecr_reduction_pct", 0.0)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="Adapted public MKT checkpoint sanity check",
        scripts=("scripts/run_openimages_mkt_baseline.py",),
        configs=("configs/openimages_10k_vitb16_heldout.yaml",),
        csv=(
            CsvExpectation(
                "results/supplementary/mkt_combined_summary.csv",
                methods=("mkt_open_vocab",),
                numeric=(
                    NumericExpectation((("method", "mkt_open_vocab"),), (("n", 12), ("average_precision_mean", 0.6263), ("best_f1_mean", 0.6626), ("tecr_mean", 0.6399)), 5e-4),
                ),
            ),
        ),
        note="Requires external MKT checkout and public checkpoint files.",
    ),
    ClaimCheck(
        name="Gate parameter stability",
        scripts=("scripts/summarize_gate_parameter_stability.py",),
        csv=(
            CsvExpectation(
                "results/supplementary/gate_parameter_stability_summary.csv",
                columns=("dataset", "residual_power_mean", "confidence_cutoff_mean"),
                numeric=(
                    NumericExpectation((("dataset", "Open Images 10k"),), (("num_selected", 60), ("residual_power_mean", 0.243), ("residual_power_std", 0.105), ("confidence_cutoff_mean", 0.442), ("confidence_cutoff_std", 0.125), ("confidence_temperature_mean", 0.053), ("confidence_temperature_std", 0.034)), 5e-4),
                    NumericExpectation((("dataset", "COCO val2017"),), (("num_selected", 60), ("residual_power_mean", 0.281), ("residual_power_std", 0.108), ("confidence_cutoff_mean", 0.522), ("confidence_cutoff_std", 0.111), ("confidence_temperature_mean", 0.041), ("confidence_temperature_std", 0.037)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="ViT-B/16 backbone check",
        scripts=(
            "scripts/run_openimages_pilot.py",
            "scripts/run_openimages_heldout_calibration.py",
        ),
        configs=(
            "configs/openimages_10k_vitb16_heldout.yaml",
            "configs/openimages_10k_vitb16_heldout_classB.yaml",
        ),
        csv=(
            CsvExpectation(
                "results/supplementary/openimages_vitb16_heldout_summary.csv",
                methods=("clip_knn", "elta_confidence_heldout"),
                numeric=(
                    NumericExpectation((("method", "clip_knn"),), (("tecr_mean", 0.2654), ("tecr_reduction_pct", 0.0)), 5e-4),
                    NumericExpectation((("method", "elta_confidence_heldout"),), (("tecr_mean", 0.2255), ("tecr_reduction_pct", 15.1)), 5e-4),
                ),
            ),
        ),
    ),
    ClaimCheck(
        name="Frequency-group diagnostics and COCO exceptions",
        scripts=(
            "scripts/analyze_openimages_heldout_groups.py",
            "scripts/summarize_fairness_diagnostics.py",
        ),
        csv=(
            CsvExpectation(
                "results/supplementary/frequency_group_12config_summary.csv",
                columns=("dataset", "frequency_group", "clip_knn_tecr_mean", "heldout_tecr_mean"),
                numeric=(
                    NumericExpectation((("dataset", "Open Images 10k"), ("frequency_group", "head")), (("num_emerging_labels_mean", 4.3), ("clip_knn_tecr_mean", 0.1446), ("heldout_tecr_mean", 0.1347), ("tecr_delta", -0.0099), ("tecr_reduction_pct", 6.8)), 5e-4),
                    NumericExpectation((("dataset", "Open Images 10k"), ("frequency_group", "mid")), (("num_emerging_labels_mean", 4.8), ("clip_knn_tecr_mean", 0.0823), ("heldout_tecr_mean", 0.0776), ("tecr_delta", -0.0046), ("tecr_reduction_pct", 5.6)), 5e-4),
                    NumericExpectation((("dataset", "Open Images 10k"), ("frequency_group", "tail")), (("num_emerging_labels_mean", 5.9), ("clip_knn_tecr_mean", 0.0419), ("heldout_tecr_mean", 0.0405), ("tecr_delta", -0.0014), ("tecr_reduction_pct", 3.2)), 5e-4),
                    NumericExpectation((("dataset", "COCO val2017"), ("frequency_group", "head")), (("num_emerging_labels_mean", 5.9), ("clip_knn_tecr_mean", 0.1947), ("heldout_tecr_mean", 0.1611), ("tecr_delta", -0.0336), ("tecr_reduction_pct", 17.3)), 5e-4),
                    NumericExpectation((("dataset", "COCO val2017"), ("frequency_group", "mid")), (("num_emerging_labels_mean", 5.3), ("clip_knn_tecr_mean", 0.0526), ("heldout_tecr_mean", 0.0453), ("tecr_delta", -0.0074), ("tecr_reduction_pct", 14.0)), 5e-4),
                    NumericExpectation((("dataset", "COCO val2017"), ("frequency_group", "tail")), (("num_emerging_labels_mean", 3.8), ("clip_knn_tecr_mean", 0.0236), ("heldout_tecr_mean", 0.0262), ("tecr_delta", 0.0027), ("tecr_reduction_pct", -11.3)), 5e-4),
                ),
            ),
            CsvExpectation(
                "results/supplementary/coco_training_exception_configs.csv",
                columns=("run_key", "best_training_method", "heldout_gate_tecr"),
                numeric=(
                    NumericExpectation((("run_key", "s20260524"),), (("best_training_tecr", 0.2701), ("heldout_gate_tecr", 0.2736), ("clip_knn_tecr", 0.3354)), 5e-4),
                    NumericExpectation((("run_key", "s20260525"),), (("best_training_tecr", 0.2304), ("heldout_gate_tecr", 0.2480), ("clip_knn_tecr", 0.3399)), 5e-4),
                ),
            ),
        ),
    ),
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values_match(actual: str | None, expected: str) -> bool:
    return actual is not None and actual.strip() == expected


def parse_number(value: str) -> float:
    return float(value)


def numeric_tolerance(expected: float, default: float) -> float:
    return max(default, 0.05 if abs(expected) >= 1.0 else default)


def check_numeric(expectation: CsvExpectation, rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for numeric in expectation.numeric:
        matched = [
            row
            for row in rows
            if all(values_match(row.get(column), expected) for column, expected in numeric.match)
        ]
        label = ", ".join(f"{column}={expected}" for column, expected in numeric.match)
        if not matched:
            errors.append(f"{expectation.path}: missing row matching {label}")
            continue
        row = matched[0]
        for column, expected in numeric.values:
            if column not in row:
                errors.append(f"{expectation.path}: row {label} missing numeric column `{column}`")
                continue
            try:
                actual = parse_number(row[column])
            except ValueError:
                errors.append(f"{expectation.path}: row {label} column `{column}` is not numeric: {row[column]!r}")
                continue
            tolerance = numeric_tolerance(expected, numeric.tolerance)
            if abs(actual - expected) > tolerance:
                errors.append(
                    f"{expectation.path}: row {label} column `{column}` expected {expected} "
                    f"within {tolerance}, found {actual}"
                )
    return errors


def check_csv(expectation: CsvExpectation) -> list[str]:
    path = ROOT / expectation.path
    errors: list[str] = []
    if not path.exists():
        return [f"missing result CSV: {expectation.path}"]
    rows = read_csv_rows(path)
    if not rows:
        return [f"empty result CSV: {expectation.path}"]
    columns = set(rows[0].keys())
    for column in expectation.columns:
        if column not in columns:
            errors.append(f"{expectation.path}: missing column `{column}`")
    if expectation.methods:
        found = {row.get("method", "") for row in rows}
        for method in expectation.methods:
            if method not in found:
                errors.append(f"{expectation.path}: missing method `{method}`")
    errors.extend(check_numeric(expectation, rows))
    return errors


def check_path(kind: str, path: str) -> str | None:
    if not (ROOT / path).exists():
        return f"missing {kind}: {path}"
    return None


def run_checks() -> list[tuple[ClaimCheck, list[str]]]:
    results: list[tuple[ClaimCheck, list[str]]] = []
    for check in CHECKS:
        errors: list[str] = []
        for path in check.scripts:
            error = check_path("script", path)
            if error:
                errors.append(error)
        for path in check.configs:
            error = check_path("config", path)
            if error:
                errors.append(error)
        for path in check.results:
            error = check_path("artifact", path)
            if error:
                errors.append(error)
        for expectation in check.csv:
            errors.extend(check_csv(expectation))
        results.append((check, errors))
    return results


def print_markdown(results: list[tuple[ClaimCheck, list[str]]]) -> None:
    print("# Reproducibility Completeness Audit\n")
    print("| Claim group | Status | Details |")
    print("|---|---|---|")
    for check, errors in results:
        status = "PASS" if not errors else "FAIL"
        details = check.note if not errors else "<br>".join(errors)
        print(f"| {check.name} | {status} | {details} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Print only failures.")
    args = parser.parse_args()

    results = run_checks()
    if args.quiet:
        for check, errors in results:
            for error in errors:
                print(f"{check.name}: {error}")
    else:
        print_markdown(results)
    return 1 if any(errors for _, errors in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
