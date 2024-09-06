import json
import os

import gradio as gr
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results(output_dir):
    results_file = os.path.join(SCRIPT_DIR, output_dir, "experiment_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return []


def display_experiment(
    output_dir,
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    use_m2m,
    min_mask_region_area,
    points_per_batch,
):
    results = load_results(output_dir)

    # Find the experiment with the selected parameters
    selected_experiment = next(
        (
            r
            for r in results
            if r["params"]["points_per_side"] == points_per_side
            and abs(r["params"]["pred_iou_thresh"] - pred_iou_thresh) < 0.01
            and abs(r["params"]["stability_score_thresh"] - stability_score_thresh) < 0.01
            and r["params"]["use_m2m"] == use_m2m
            and r["params"]["min_mask_region_area"] == min_mask_region_area
            and r["params"]["points_per_batch"] == points_per_batch
        ),
        None,
    )

    if not selected_experiment:
        return None, "No experiment found for the selected parameters"

    # Load and display visualization
    vis_filename = selected_experiment["visualization_filename"]
    vis_path = os.path.join(SCRIPT_DIR, output_dir, selected_experiment["experiment_id"], vis_filename)

    if os.path.exists(vis_path):
        vis_image = Image.open(vis_path)
    else:
        return None, f"Visualization not found at {vis_path}"

    # Prepare experiment details
    details = f"Experiment ID: {selected_experiment['experiment_id']}\n"
    details += "Parameters:\n"
    for key, value in selected_experiment["params"].items():
        details += f"  {key}: {value}\n"
    details += f"Number of masks: {len(selected_experiment['mask_filenames'])}\n"

    return vis_image, details


# Set up the Gradio interface
output_dir = "experiment_output"  # This is now relative to the script location

with gr.Blocks() as demo:
    gr.Markdown("# SAM2 Mask Generation Experiment Viewer")

    with gr.Row():
        output_dir_input = gr.Textbox(label="Output Directory", value=output_dir)
        points_per_side_input = gr.Dropdown(choices=[16, 32, 64], label="Points per Side", value=32)
        pred_iou_thresh_input = gr.Dropdown(choices=[0.8, 0.85, 0.9], label="Predicted IoU Threshold", value=0.8)
        stability_score_thresh_input = gr.Dropdown(
            choices=[0.85, 0.9, 0.95], label="Stability Score Threshold", value=0.9
        )

    with gr.Row():
        use_m2m_input = gr.Checkbox(label="Use Mask-to-Mask Refinement", value=False)
        min_mask_region_area_input = gr.Dropdown(choices=[0, 100, 1000], label="Minimum Mask Region Area", value=0)
        points_per_batch_input = gr.Dropdown(choices=[32, 64, 128], label="Points per Batch", value=64)

    with gr.Row():
        vis_image = gr.Image(label="Visualization")

    experiment_details = gr.Textbox(label="Experiment Details", lines=10)

    # Set up interactions
    inputs = [
        output_dir_input,
        points_per_side_input,
        pred_iou_thresh_input,
        stability_score_thresh_input,
        use_m2m_input,
        min_mask_region_area_input,
        points_per_batch_input,
    ]
    outputs = [vis_image, experiment_details]

    # Update visualization when any input changes
    for input_component in inputs:
        input_component.change(display_experiment, inputs=inputs, outputs=outputs)

# Launch the app
if __name__ == "__main__":
    demo.launch()
