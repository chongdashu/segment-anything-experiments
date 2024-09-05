import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2


def find_checkpoint_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    checkpoint_path = os.path.join(project_root, "segment-anything-2", "checkpoints", "sam2_hiera_large.pt")

    if not os.path.exists(checkpoint_path):
        # Try alternative path
        checkpoint_path = os.path.join(project_root, "checkpoints", "sam2_hiera_large.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}")

    return checkpoint_path


def load_sam2_model():
    checkpoint = find_checkpoint_path()
    model_cfg = "sam2_hiera_l.yaml"

    try:
        sam2_model = build_sam2(model_cfg, checkpoint)
        print(f"Successfully loaded SAM2 model from: {checkpoint}")
        return sam2_model
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        print("Please ensure the checkpoint file is in the correct location.")
        raise


def load_image(filename="input.png"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tutorial_root = os.path.dirname(current_dir)
    image_path = os.path.join(tutorial_root, filename)
    if os.path.exists(image_path):
        return np.array(Image.open(image_path))
    image_path = os.path.join(current_dir, filename)
    if os.path.exists(image_path):
        return np.array(Image.open(image_path))
    raise FileNotFoundError(f"Could not find {filename} in the tutorial root or current directory.")


def plot_masks(image, masks, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        show_mask(mask, plt.gca(), random_color=True)

        # Find the center of the mask
        y, x = np.where(mask)
        if len(y) > 0 and len(x) > 0:
            center_y, center_x = int(np.mean(y)), int(np.mean(x))

            # Add a white circle as background for the text
            circle = patches.Circle((center_x, center_y), radius=10, fill=True, color="white", alpha=0.7)
            plt.gca().add_patch(circle)

            # Add the index number
            plt.text(center_x, center_y, str(i), color="black", fontsize=8, ha="center", va="center", fontweight="bold")

    plt.axis("off")
    return plt.gcf()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_mask(mask, filename, output_dir):
    """Saves a single mask as a PNG file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Convert boolean mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    cv2.imwrite(filepath, mask_uint8)
    print(f"Saved mask to {filepath}")


def save_output(fig, filename, output_dir):
    """Saves the figure without extra whitespace."""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved figure to {filepath}")
