from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_video_predictor


def find_checkpoint_path():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    checkpoint_path = project_root.joinpath("segment-anything-2", "checkpoints", "sam2_hiera_large.pt")

    if not checkpoint_path.exists():
        checkpoint_path = project_root.joinpath("checkpoints", "sam2_hiera_large.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}")

    return checkpoint_path


def load_sam2_model():
    checkpoint = find_checkpoint_path()
    model_cfg = "sam2_hiera_l.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
        print(f"Successfully loaded SAM2 model from: {checkpoint}")
        return predictor
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        print("Please ensure the checkpoint file is in the correct location.")
        raise


def ensure_rgb(image):
    if image.shape[2] == 4:
        print("Converting RGBA image to RGB")
        return image[:, :, :3]
    return image


def extract_frames(video_path, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_path = output_folder.joinpath(f"{frame_count:05d}.jpg")
        cv2.imwrite(str(frame_path), frame)
        frame_count += 1
    video.release()
    return frame_count


def process_video(predictor, video_folder, object_points, object_labels):
    inference_state = predictor.init_state(video_path=str(video_folder))

    frame_idx = 0
    obj_id = 1
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=object_points,
        labels=object_labels,
    )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def remove_object_from_frame(frame, mask, inpaint_radius=5):
    # Ensure frame is in BGR format (3 channels)
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Ensure mask is a single-channel (1 channel) uint8 image
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Check that mask is single-channel
    if len(mask_uint8.shape) > 2:
        mask_uint8 = mask_uint8[:, :, 0]

    # Resize mask to match the frame size if necessary
    if mask_uint8.shape[:2] != frame.shape[:2]:
        mask_uint8 = cv2.resize(mask_uint8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Perform inpainting with adjusted radius
    frame_without_object = cv2.inpaint(frame, mask_uint8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    return frame_without_object


def visualize_mask_on_frame(frame, mask):
    plt.figure(figsize=(10, 6))
    plt.title("Mask Overlay on Frame")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, cmap="jet", alpha=0.5)  # Overlay mask in jet color
    plt.axis("off")
    plt.show()


def create_energy_trail_frames(video_segments, input_folder, output_folder, frame_count, trail_length=15):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    recent_masks = []

    # Create a larger kernel for a more pronounced glow effect
    glow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for i in range(frame_count):
        frame_path = input_folder.joinpath(f"{i:05d}.jpg")
        frame = cv2.imread(str(frame_path))

        if i in video_segments:
            mask = video_segments[i][1]
            mask = np.squeeze(mask).astype(np.float32)

            # Add current mask to recent masks
            recent_masks.append(mask)
            if len(recent_masks) > trail_length:
                recent_masks.pop(0)

            # Create energy trail effect
            trail_mask = np.zeros_like(mask)
            for j, old_mask in enumerate(recent_masks):
                weight = (j + 1) / len(recent_masks)
                trail_mask += old_mask * weight

            # Normalize trail mask
            trail_mask = cv2.normalize(trail_mask, None, 0, 1, cv2.NORM_MINMAX)

            # Create glow effect
            glow_mask = cv2.dilate(trail_mask, glow_kernel)
            glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 0)

            # Apply energy trail effect to frame
            energy_color = np.array([255, 100, 0], dtype=np.float32)  # Blue color
            glow_frame = frame.copy().astype(np.float32)

            for c in range(3):
                glow_frame[:, :, c] = frame[:, :, c] * (1 - glow_mask) + energy_color[c] * glow_mask

            # Blend energy frame with original frame
            frame = cv2.addWeighted(frame, 1, glow_frame.astype(np.uint8), 0.7, 0)

        output_frame_path = output_folder.joinpath(f"{i:04d}.png")
        cv2.imwrite(str(output_frame_path), frame)

    print(f"Frames with energy trail effect saved to {output_folder}")


def create_ghost_ball_frames(video_segments, input_folder, output_folder, frame_count, opacity=0.5, trail_length=10):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    recent_masks = []

    for i in range(frame_count):
        frame_path = input_folder.joinpath(f"{i:05d}.jpg")
        frame = cv2.imread(str(frame_path))

        if i in video_segments:
            mask = video_segments[i][1]
            mask = np.squeeze(mask).astype(np.float32)

            recent_masks.append(mask)
            if len(recent_masks) > trail_length:
                recent_masks.pop(0)

            ghost_mask = np.zeros_like(mask)
            for j, old_mask in enumerate(recent_masks):
                ghost_mask += old_mask * (opacity ** (len(recent_masks) - j - 1))

            ghost_mask = np.clip(ghost_mask, 0, 1)

            # Apply ghost effect to frame
            ghost_frame = frame.copy().astype(np.float32)
            yellow_tint = np.array([0, 255, 255], dtype=np.float32)

            mask_3d = np.repeat(ghost_mask[:, :, np.newaxis], 3, axis=2)
            ghost_frame = ghost_frame * (1 - mask_3d * opacity) + yellow_tint * (mask_3d * opacity)

            frame = cv2.addWeighted(frame, 1, ghost_frame.astype(np.uint8), 1, 0)

        output_frame_path = output_folder.joinpath(f"{i:04d}.png")
        cv2.imwrite(str(output_frame_path), frame)

    print(f"Frames with ghost ball effect saved to {output_folder}")


def create_output_frames(video_segments, input_folder, output_folder, frame_count):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i in range(frame_count):
        frame_path = input_folder.joinpath(f"{i:05d}.jpg")
        frame = Image.open(str(frame_path))

        if i in video_segments:
            mask = video_segments[i][1]

            # Debug: Print mask shape and unique values
            print(f"Frame {i}: Mask shape: {mask.shape}, Unique values: {np.unique(mask)}")

            # Remove singleton dimensions if present
            mask = np.squeeze(mask)

            # Find bounding box of the mask
            mask_binary = mask > 0.0
            rows = np.any(mask_binary, axis=1)
            cols = np.any(mask_binary, axis=0)

            if rows.any() and cols.any():  # Check if any True values in rows and cols
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                # Debug: Print bounding box coordinates
                print(f"Frame {i}: Bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

                # Ensure x_max > x_min and y_max > y_min
                if x_max > x_min and y_max > y_min:
                    # Draw bounding box
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                else:
                    print(f"Frame {i}: Invalid bounding box coordinates")
            else:
                print(f"Frame {i}: No object detected in mask")

        output_frame_path = output_folder.joinpath(f"{i:04d}.png")
        frame.save(str(output_frame_path))

    print(f"Frames with bounding boxes saved to {output_folder}")


def create_erased_output_frames(video_segments, input_folder, output_folder, frame_count):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i in range(frame_count):
        frame_path = input_folder.joinpath(f"{i:05d}.jpg")
        frame = cv2.imread(str(frame_path))

        if i in video_segments:
            mask = video_segments[i][1]

            # Debug: Print mask shape and unique values
            print(f"Frame {i}: Mask shape: {mask.shape}, Unique values: {np.unique(mask)}")

            # Remove singleton dimensions if present
            mask = np.squeeze(mask)

            # Erase the object from the frame
            frame_without_object = remove_object_from_frame(frame, mask)
        else:
            frame_without_object = frame

        output_frame_path = output_folder.joinpath(f"{i:04d}.png")
        cv2.imwrite(str(output_frame_path), frame_without_object)

    print(f"Frames with erased objects saved to {output_folder}")


def create_output_video(output_frames_dir, output_video_path, frame_rate=30):
    output_frames_dir = Path(output_frames_dir)
    output_video_path = Path(output_video_path)

    # Get the list of frame files
    frame_files = sorted(output_frames_dir.glob("*.png"))

    if not frame_files:
        print("No frames found in the output directory.")
        return

    # Read the first frame to get the frame size
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, frame_rate, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Output video saved to: {output_video_path}")
