import gradio as gr
from common import load_experiment_result, load_image


def display_experiment_result(points_per_side, pred_iou_thresh, stability_score_thresh, **kwargs):
    params = {
        "points_per_side": points_per_side,
        "pred_iou_thresh": pred_iou_thresh,
        "stability_score_thresh": stability_score_thresh,
        **kwargs,
    }
    result = load_experiment_result("experiment_output", params)

    if result and result["visualization_path"]:
        return result["visualization_path"]
    else:
        return None


def load_original_image():
    return load_image("input.png")


with gr.Blocks(title="SAM2 Automatic Mask Generation Experiments") as demo:
    gr.Markdown("## SAM2 Automatic Mask Generation Experiments")
    gr.Markdown("Adjust parameters to see different mask generation results.")

    with gr.Row():
        with gr.Column(scale=1):
            points_per_side = gr.Slider(16, 64, step=16, value=32, label="Points per side")
            pred_iou_thresh = gr.Slider(0.5, 1.0, step=0.1, value=0.8, label="Pred IoU Threshold")
            stability_score_thresh = gr.Slider(0.5, 1.0, step=0.05, value=0.9, label="Stability Score Threshold")

            # Additional parameters (set to default values)
            points_per_batch = gr.Slider(32, 128, step=32, value=64, label="Points per batch")
            stability_score_offset = gr.Slider(0.0, 2.0, step=0.1, value=1.0, label="Stability Score Offset")
            box_nms_thresh = gr.Slider(0.0, 1.0, step=0.1, value=0.7, label="Box NMS Threshold")
            crop_n_layers = gr.Slider(0, 5, step=1, value=0, label="Crop N Layers")
            crop_nms_thresh = gr.Slider(0.0, 1.0, step=0.1, value=0.7, label="Crop NMS Threshold")
            crop_overlap_ratio = gr.Slider(0.0, 1.0, step=0.1, value=512 / 1500, label="Crop Overlap Ratio")
            crop_n_points_downscale_factor = gr.Slider(1, 5, step=1, value=1, label="Crop N Points Downscale Factor")
            min_mask_region_area = gr.Slider(0, 1000, step=100, value=0, label="Min Mask Region Area")
            output_mode = gr.Radio(
                ["binary_mask", "uncompressed_rle", "coco_rle"], value="binary_mask", label="Output Mode"
            )
            use_m2m = gr.Checkbox(value=False, label="Use M2M")
            multimask_output = gr.Checkbox(value=True, label="Multimask Output")

            submit_button = gr.Button("Generate Masks")

        with gr.Column(scale=2):
            with gr.Row():
                original_image = gr.Image(label="Original Image")
                result_image = gr.Image(label="Generated Masks")

    submit_button.click(
        fn=display_experiment_result,
        inputs=[
            points_per_side,
            pred_iou_thresh,
            stability_score_thresh,
            points_per_batch,
            stability_score_offset,
            box_nms_thresh,
            crop_n_layers,
            crop_nms_thresh,
            crop_overlap_ratio,
            crop_n_points_downscale_factor,
            min_mask_region_area,
            output_mode,
            use_m2m,
            multimask_output,
        ],
        outputs=result_image,
    )

    demo.load(fn=load_original_image, outputs=original_image)

if __name__ == "__main__":
    demo.launch()
