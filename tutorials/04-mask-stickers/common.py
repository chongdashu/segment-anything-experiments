# common.py

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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

def load_image(filename='input.png'):
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
    for mask, score in zip(masks, scores):
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis('off')
    return plt.gcf()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def create_sticker(image, selected_masks):
    combined_mask = np.logical_or.reduce(selected_masks)
    sticker = np.zeros_like(image)
    sticker[combined_mask] = image[combined_mask]
    return Image.fromarray(sticker)

def save_output(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    print(f"Saved {filename} to {output_dir}")

def save_sticker(sticker, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sticker.save(os.path.join(output_dir, filename))
    print(f"Saved {filename} to {output_dir}")
