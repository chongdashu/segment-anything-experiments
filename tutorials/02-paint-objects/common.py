import os

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def find_checkpoint_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', ".."))
    checkpoint_path = os.path.join(project_root, 'segment-anything-2', 'checkpoints', 'sam2_hiera_large.pt')
    
    if not os.path.exists(checkpoint_path):
        # Try alternative path
        checkpoint_path = os.path.join(project_root, 'checkpoints', 'sam2_hiera_large.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}")
    
    return checkpoint_path

# Initialize SAM2 model
checkpoint = find_checkpoint_path()
model_cfg = "sam2_hiera_l.yaml"

try:
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    print(f"Successfully loaded SAM2 model from: {checkpoint}")
except Exception as e:
    print(f"Error loading SAM2 model: {e}")
    print("Please ensure the checkpoint file is in the correct location.")
    raise

def ensure_rgb(image):
    if image.shape[2] == 4:  # RGBA image
        print("Converting RGBA image to RGB")
        return image[:,:,:3]
    return image

def remove_background_points(image, points, labels):
    # Ensure image is RGB
    image = ensure_rgb(image)
    
    # Convert inputs to numpy arrays
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Set image and predict mask
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
    
    # Apply mask to original image
    mask = masks[0].astype(np.uint8) * 255
    mask_image = Image.fromarray(mask)
    image_pil = Image.fromarray(image)
    processed_image = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    processed_image.paste(image_pil, mask=mask_image)
    
    # Create comparison image
    comparison = Image.new('RGB', (image.shape[1] * 2, image.shape[0]))
    comparison.paste(Image.fromarray(image), (0, 0))
    comparison.paste(processed_image, (image.shape[1], 0))
    
    return processed_image, comparison

def remove_background_box(image, x1, y1, x2, y2):
    # Ensure image is RGB
    image = ensure_rgb(image)
    
    # Prepare input
    input_box = np.array([x1, y1, x2, y2])
    
    # Set image and predict mask
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
    
    # Apply mask to original image
    mask = masks[0].astype(np.uint8) * 255
    mask_image = Image.fromarray(mask)
    image_pil = Image.fromarray(image)
    processed_image = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    processed_image.paste(image_pil, mask=mask_image)
    
    # Create comparison image
    comparison = Image.new('RGB', (image.shape[1] * 2, image.shape[0]))
    comparison.paste(Image.fromarray(image), (0, 0))
    comparison.paste(processed_image, (image.shape[1], 0))
    
    return processed_image, comparison