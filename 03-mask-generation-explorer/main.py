import json
import os
from itertools import product

from common import load_image, load_sam2_model
from experiments import run_mask_generation_experiment, save_experiment_results


def generate_experiments():
    # Define parameter ranges for the ones we want to vary
    points_per_side_range = [16, 32, 64]
    pred_iou_thresh_range = [0.8, 0.85, 0.9]
    stability_score_thresh_range = [0.85, 0.9, 0.95]
    # pred_iou_thresh_range = [0.7, 0.8, 0.9]
    # stability_score_thresh_range = [0.85, 0.9, 0.95]

    # Define default values for other parameters
    default_params = {
        "points_per_batch": 64,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.95,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 512 / 1500,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask",
        "use_m2m": False,
        "multimask_output": True,
    }

    # Generate all combinations of the parameters we're varying
    all_combinations = list(product(points_per_side_range, pred_iou_thresh_range, stability_score_thresh_range))

    # Create experiments list
    experiments = []
    for combo in all_combinations:
        experiment = default_params.copy()
        experiment.update(
            {"points_per_side": combo[0], "pred_iou_thresh": combo[1], "stability_score_thresh": combo[2]}
        )
        experiments.append(experiment)

    return experiments


def load_existing_experiments(output_dir):
    results_file = os.path.join(output_dir, "experiment_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return []


def main():
    # Load model and image
    sam2_model = load_sam2_model()
    image = load_image("input.jpg")

    # Define output directory
    output_dir = "experiment_output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate experiments
    experiments = generate_experiments()

    # Load existing experiments
    existing_results = load_existing_experiments(output_dir)
    existing_params = [result["params"] for result in existing_results]

    # Run experiments and collect results
    results = existing_results
    for params in experiments:
        if params in existing_params:
            print(f"Warning: Experiment with params {params} already exists. Skipping.")
            continue

        print(f"Running experiment with params: {params}")
        result = run_mask_generation_experiment(image, sam2_model, params, output_dir)
        results.append(result)

        # Save results after each experiment
        save_experiment_results(results, output_dir)


if __name__ == "__main__":
    main()
