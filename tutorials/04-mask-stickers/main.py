# main.py

import os

from common import (
    create_sticker,
    load_image,
    load_sam2_model,
    plot_masks,
    save_output,
    save_sticker,
)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    # Set up output directory
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load the SAM2 model
    sam2_model = load_sam2_model()

    # Load the image
    image = load_image("cake.jpg")

    # Create the mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    # Generate masks
    masks = mask_generator.generate(image)

    # Plot masks with indices
    fig = plot_masks(image, [mask["segmentation"] for mask in masks], [mask["predicted_iou"] for mask in masks])

    # Save the plot
    save_output(fig, "detected_masks.png", output_dir)

    print(f"Detected {len(masks)} objects in the image.")
    print("Masks visualization with indices saved as 'detected_masks.png'")

    # Print information about each mask
    for i, mask in enumerate(masks):
        print(f"Mask {i}:")
        print(f"  Predicted IOU: {mask['predicted_iou']:.2f}")
        print(f"  Stability Score: {mask['stability_score']:.2f}")
        print(f"  Area: {mask['area']}")

    # Simulate user selection (in a real app, this would be interactive)
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 19, 20, 21]
    selected_masks = [masks[i]["segmentation"] for i in selected_indices]

    print(f"\nSelected mask indices: {selected_indices}")

    # Create sticker
    sticker = create_sticker(image, selected_masks)
    save_sticker(sticker, "sticker.png", output_dir)
    print("Sticker created and saved as 'sticker.png'")


if __name__ == "__main__":
    main()
