import json
import os

from common import plot_masks, save_mask, save_output

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def run_mask_generation_experiment(image, sam2_model, params, output_dir):
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, **params)
    masks = mask_generator.generate(image)

    # Create a unique identifier for this experiment
    experiment_id = f"exp_{abs(hash(json.dumps(params, sort_keys=True)))}"
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    masks_dir = os.path.join(experiment_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Save masks
    mask_filenames = []
    for i, mask_data in enumerate(masks):
        mask_filename = f"mask_{i}.png"
        save_mask(mask_data["segmentation"], mask_filename, masks_dir)
        mask_filenames.append(os.path.join("masks", mask_filename))

    # Save visualization
    fig = plot_masks(
        image, [mask_data["segmentation"] for mask_data in masks], [mask_data["predicted_iou"] for mask_data in masks]
    )
    vis_filename = "visualization.png"
    save_output(fig, vis_filename, experiment_dir)

    return {
        "params": params,
        "mask_filenames": mask_filenames,
        "visualization_filename": vis_filename,
        "experiment_id": experiment_id,
    }


def save_experiment_results(results: list[dict], output_dir: str):
    output_file = os.path.join(output_dir, "experiment_results.json")

    # Load existing results if the file exists
    existing_results = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_results = json.load(f)

    # Append new results
    updated_results = existing_results + results

    # Save updated results
    with open(output_file, "w") as f:
        json.dump(updated_results, f, indent=2)
    print(f"Saved experiment results to {output_file}")


def load_experiment_result(output_dir: str, params: dict) -> dict | None:
    results_file = os.path.join(output_dir, "experiment_results.json")
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        all_results = json.load(f)

    for result in all_results:
        if all(result["params"].get(k) == v for k, v in params.items()):
            experiment_dir = os.path.join(output_dir, result["experiment_id"])

            # Check if visualization file exists
            vis_path = os.path.join(experiment_dir, result["visualization_filename"])
            if os.path.exists(vis_path):
                result["visualization_path"] = vis_path
            else:
                print(f"Visualization file not found: {vis_path}")
                result["visualization_path"] = None

            # Check if mask files exist
            result["mask_paths"] = []
            for mask_filename in result["mask_filenames"]:
                mask_path = os.path.join(experiment_dir, mask_filename)
                if os.path.exists(mask_path):
                    result["mask_paths"].append(mask_path)
                else:
                    print(f"Mask file not found: {mask_path}")

            return result

    print(f"No matching experiment found for parameters: {params}")
    return None
