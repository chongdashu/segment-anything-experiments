import os

import numpy as np
from common import create_sticker, load_image, load_sam2_model, plot_masks, save_mask, save_output, save_sticker

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    # Set up output directory
    output_dir = os.path.join(os.getcwd(), "output")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Load the SAM2 model
    sam2_model = load_sam2_model()

    # Load the image
    image = load_image("breakfast.jpg")

    # Create the mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=20,  # Increased to capture more details
        pred_iou_thresh=0.94,  # Slightly reduced to include more masks
        stability_score_thresh=0.95,  # Slightly reduced to include more stable masks
        crop_n_layers=2,  # No cropping to prevent additional masks
        min_mask_region_area=100,  # Reduced to capture smaller elements
        box_nms_thresh=0.95,  # Keeping it strict to reduce overlap
        crop_overlap_ratio=0.3,  # Keeping the same
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
        # Create a blank mask image
        mask_image = np.zeros_like(image)

        # Apply the mask to the blank image
        mask_image[mask["segmentation"]] = image[mask["segmentation"]]

        # Save the mask
        save_mask(mask_image, f"{i}.png", masks_dir)

        print(f"Mask {i}:")
        print(f"  Predicted IOU: {mask['predicted_iou']:.2f}")
        print(f"  Stability Score: {mask['stability_score']:.2f}")
        print(f"  Area: {mask['area']}")

    # Simulate user selection (in a real app, this would be interactive)
    selected_indices = list(range(1, 14)) + [22, 23]
    selected_masks = [masks[i]["segmentation"] for i in selected_indices]

    print(f"\nSelected mask indices: {selected_indices}")

    # Create sticker
    sticker = create_sticker(image, selected_masks)
    save_sticker(sticker, "sticker.png", output_dir)
    print("Sticker created and saved as 'sticker.png'")


if __name__ == "__main__":
    main()
