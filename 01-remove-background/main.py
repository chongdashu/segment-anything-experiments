import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load the image
image = cv2.imread("input.png")

if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Specify the checkpoint and model configuration
checkpoint = "../segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Build the model
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Refined input prompts
input_prompts = {
    "point_coords": np.array(
        [
            [150, 200],  # Cat's body (positive)
            [320, 180],  # Cat's tail (positive)
            [60, 100],  # Background (negative)
            [450, 350],  # Background (negative)
            [100, 50],  # Background (negative)
            [200, 400],  # Cat's leg (positive)
            [400, 150],  # Background (negative)
        ]
    ),
    "point_labels": np.array([1, 1, 0, 0, 0, 1, 0]),  # 1 for object, 0 for background
}

# Set the image and predict the mask
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=input_prompts["point_coords"], point_labels=input_prompts["point_labels"]
    )

for index, mask in enumerate(masks):
    # Save the mask as a separate PNG file
    cv2.imwrite(f"mask{index}.png", (mask * 255).astype(np.uint8))

    # Ensure `background_mask` is binary and convert to uint8
    background_mask = (mask > 0).astype(np.uint8) * 255  # Convert to 0 and 255

    # Invert the mask if necessary (depends on whether you want to keep the cat or remove it)
    # inverted_mask = cv2.bitwise_not(background_mask)  # Uncomment if you need to invert

    # Apply the mask to the image to remove the background
    background_removed_image = cv2.bitwise_and(image, image, mask=background_mask)

    # Create an output image where the background is transparent (for PNG output)
    output_image = np.dstack([background_removed_image, background_mask])

    # Save the result as PNG to preserve transparency
    cv2.imwrite(f"output{index}.png", output_image)
