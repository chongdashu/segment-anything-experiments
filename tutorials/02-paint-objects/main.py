import gradio as gr
import numpy as np
from common import predictor, remove_background_box
from PIL import Image


def ensure_rgb(image):
    if image.shape[-1] == 4:  # RGBA image
        return image[:, :, :3]
    return image


def create_transparent_mask(mask, color=(255, 0, 0, 128)):
    """Create a transparent overlay from the mask"""
    rgba_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba_mask[mask > 128] = color
    return rgba_mask


def remove_background(image):
    if image is None or "background" not in image or image["background"] is None:
        return None, None, "No image uploaded."

    original_image = np.array(image["background"])
    original_image_rgb = ensure_rgb(original_image)
    height, width = original_image_rgb.shape[:2]
    x1, y1 = width * 0.1, height * 0.1
    x2, y2 = width * 0.9, height * 0.9

    processed_image, _ = remove_background_box(original_image_rgb, x1, y1, x2, y2)
    mask = np.array(processed_image)[:, :, 3]  # Extract alpha channel as mask
    transparent_mask = create_transparent_mask(mask)

    editor_output = {
        "background": original_image,
        "layers": [original_image, transparent_mask],  # Layer 1: original, Layer 2: transparent mask
        "composite": None,
    }

    return (
        editor_output,
        np.array(processed_image),
        "Background removed. Use the brush to refine the selection if needed.",
    )


def process_image(image):
    if image is None or "background" not in image or image["background"] is None:
        return None, "No image uploaded."

    original_image = np.array(image["background"])
    original_image_rgb = ensure_rgb(original_image)

    if "layers" in image and len(image["layers"]) > 1:
        mask_layer = np.array(image["layers"][1])
        if mask_layer.ndim == 3 and mask_layer.shape[2] == 4:
            mask = mask_layer[:, :, 3]  # Use alpha channel of Layer 2 as mask
        else:
            mask = mask_layer
    else:
        return None, "No mask detected. Please use the brush to mark the object."

    # Convert mask to PIL Image and resize
    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize((256, 256), Image.BILINEAR)

    # Convert back to numpy array and normalize to 0-1 range
    mask_input = np.array(mask_resized).astype(np.float32) / 255.0

    # Check if the mask is empty
    if mask_input.sum() == 0:
        return None, "No mask detected. Please use the brush to mark the object."

    # Set the image in the predictor
    predictor.set_image(original_image_rgb)

    # Predict using the mask input
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=mask_input[None, :, :],  # Add batch dimension
        multimask_output=False,
        return_logits=False,
    )

    # Apply the new mask to the original image
    new_mask = masks[0].astype(np.uint8) * 255
    mask_image = Image.fromarray(new_mask, mode="L")
    image_pil = Image.fromarray(original_image)
    result_image = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    result_image.paste(image_pil, mask=mask_image)

    return np.array(result_image), "Image processed successfully."


def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive Object Segmentation")
        gr.Markdown("1. Upload an image.")
        gr.Markdown("2. Click 'Remove Background' to create an initial mask in Layer 2.")
        gr.Markdown("3. Use the brush to refine the mask in Layer 2.")
        gr.Markdown("4. Click 'Process Image' to update the result based on the current mask.")

        with gr.Row():
            editor = gr.ImageEditor(
                label="Image Editor", brush=gr.Brush(default_size=20, colors=["#ff0000"]), height=500
            )
            output_image = gr.Image(label="Processed Image")

        status_text = gr.Textbox(label="Status", interactive=False)

        remove_bg_btn = gr.Button("Remove Background")
        process_btn = gr.Button("Process Image")

        remove_bg_btn.click(fn=remove_background, inputs=editor, outputs=[editor, output_image, status_text])

        process_btn.click(fn=process_image, inputs=editor, outputs=[output_image, status_text])

    return demo


if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
