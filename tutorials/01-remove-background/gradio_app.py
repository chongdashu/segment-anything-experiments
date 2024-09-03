import gradio as gr
import numpy as np
from common import remove_background_box, remove_background_points


def process_points(image_data):
    points = []
    labels = []

    # Ensure layers exist and contain brush strokes
    if "layers" in image_data and image_data["layers"]:
        for layer in image_data["layers"]:
            if isinstance(layer, np.ndarray):
                # This is a mask layer
                mask = layer.astype(bool)
                coords = np.array(np.where(mask)).T
                points.extend(coords[:, [1, 0]])  # Swap x and y coordinates
                labels.extend([1] * coords.shape[0])  # Assuming all points are foreground

    if not points:
        print("No points to process")
        return image_data["background"], image_data["background"]  # No points to process

    processed_image, comparison = remove_background_points(image_data["background"], points, labels)
    return processed_image, comparison


def process_box(image_data):
    if "layers" in image_data and len(image_data["layers"]) > 0:
        # Assuming the first layer contains the bounding box
        mask = image_data["layers"][0]
        if isinstance(mask, np.ndarray):
            coords = np.array(np.where(mask)).T
            if coords.size > 0:
                y1, x1 = np.min(coords, axis=0)
                y2, x2 = np.max(coords, axis=0)
                processed_image, comparison = remove_background_box(image_data["background"], x1, y1, x2, y2)
                return processed_image, comparison
    return image_data["background"], image_data["background"]  # If no box is drawn, return original image


def create_gradio_app():
    # Interface for points-based background removal
    iface_points = gr.Interface(
        fn=process_points,
        inputs=[
            gr.ImageEditor(
                type="numpy", label="Upload Product Image and Place Points", brush=gr.Brush(colors=["#00FF00"])
            )
        ],
        outputs=[gr.Image(type="pil", label="Processed Image"), gr.Image(type="pil", label="Before/After Comparison")],
        title="Product Background Removal with Points",
        description="Upload an image and place points to remove bg. Use the green brush to mark foreground areas.",
    )

    # Interface for box-based background removal
    iface_box = gr.Interface(
        fn=process_box,
        inputs=[
            gr.ImageEditor(
                type="numpy", label="Upload Product Image and Draw a Box", brush=gr.Brush(colors=["#00FF00"])
            )
        ],
        outputs=[gr.Image(type="pil", label="Processed Image"), gr.Image(type="pil", label="Before/After Comparison")],
        title="Product Background Removal with Box",
        description="Upload an image and draw a bounding box to remove the background.",
    )

    return gr.TabbedInterface([iface_points, iface_box], ["Points-based", "Box-based"])


demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch()
