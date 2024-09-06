import json
import os

import gradio as gr
import numpy as np
from experiments import load_experiment_result
from PIL import Image

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results(output_dir):
    results_file = os.path.join(SCRIPT_DIR, output_dir, "experiment_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return []


def display_experiment(output_dir, selected_experiment):
    if not selected_experiment:
        return None, None, "No experiment selected"

    params = json.loads(selected_experiment)
    result = load_experiment_result(os.path.join(SCRIPT_DIR, output_dir), params)

    if result is None:
        return None, None, "Experiment result not found"

    # Load and display visualization
    vis_path = os.path.join(SCRIPT_DIR, result["visualization_path"])
    if vis_path and os.path.exists(vis_path):
        vis_image = Image.open(vis_path)
    else:
        vis_image = None

    # Load and display first mask
    if result["mask_paths"]:
        mask_path = os.path.join(SCRIPT_DIR, result["mask_paths"][0])
        mask = np.array(Image.open(mask_path))

        # Handle different mask data types
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype == np.uint8:
            mask = mask
        else:
            mask = (mask > 0).astype(np.uint8) * 255

        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(2)

        mask_image = Image.fromarray(mask).convert("RGB")
    else:
        mask_image = None

    # Prepare experiment details
    details = f"Experiment ID: {result['experiment_id']}\n"
    details += "Parameters:\n"
    for key, value in result["params"].items():
        details += f"  {key}: {value}\n"
    details += f"Number of masks: {len(result['mask_paths'])}\n"

    return vis_image, mask_image, details


def update_experiment_list(output_dir):
    results = load_results(output_dir)
    return gr.Dropdown.update(choices=[json.dumps(r["params"]) for r in results])


# Set up the Gradio interface
output_dir = "experiment_output"  # This is now relative to the script location

with gr.Blocks() as demo:
    gr.Markdown("# SAM2 Mask Generation Experiment Viewer")

    with gr.Row():
        output_dir_input = gr.Textbox(label="Output Directory", value=output_dir)
        refresh_button = gr.Button("Refresh Experiments")

    experiment_dropdown = gr.Dropdown(choices=[], label="Select Experiment")

    with gr.Row():
        vis_image = gr.Image(label="Visualization")
        mask_image = gr.Image(label="Sample Mask")

    experiment_details = gr.Textbox(label="Experiment Details", lines=10)

    # Set up interactions
    refresh_button.click(update_experiment_list, inputs=[output_dir_input], outputs=[experiment_dropdown])

    experiment_dropdown.change(
        display_experiment,
        inputs=[output_dir_input, experiment_dropdown],
        outputs=[vis_image, mask_image, experiment_details],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
