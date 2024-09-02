from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import numpy as np
from common import (
    create_output_video,
    create_output_frames,
    extract_frames,
    load_sam2_model,
    process_video,
    remove_object_from_frame,
)


def visualize_results(video_segments, temp_folder, frame_indices):
    for idx in frame_indices:
        frame_path = temp_folder.joinpath(f"{idx:05d}.jpg")
        frame = plt.imread(str(frame_path))

        # Remove any singleton dimensions (like the extra dimension in (1, 1440, 2560))
        frame = np.squeeze(frame)

        mask = video_segments[idx][1] if idx in video_segments else None

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(frame)
        plt.title("Original Frame")
        plt.axis("off")

        if mask is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Segmentation Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            removed = remove_object_from_frame(frame, mask)
            plt.imshow(removed)
            plt.title("Object Removed")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    for idx in frame_indices:
        frame_path = temp_folder.joinpath(f"{idx:05d}.jpg")
        frame = plt.imread(str(frame_path))

        mask = video_segments[idx][1] if idx in video_segments else None

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(frame)
        plt.title("Original Frame")
        plt.axis("off")

        if mask is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Segmentation Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            removed = remove_object_from_frame(frame, mask)
            plt.imshow(removed)
            plt.title("Object Removed")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


from pathlib import Path
import numpy as np
from common import (
    create_output_frames,
    extract_frames,
    load_sam2_model,
    process_video,
)

def main():
    script_dir = Path(__file__).resolve().parent
    video_path = script_dir.joinpath("videos", "basketball-short.mp4")
    temp_folder = script_dir.joinpath("temp_frames")
    output_frames = script_dir.joinpath("output", "frames")

    predictor = load_sam2_model()

    # Extract frames
    frame_count = extract_frames(str(video_path), str(temp_folder))

    # Manually set the initial position of the basketball
    initial_position = np.array([[550, 150]], dtype=np.float32)

    # Define object points and labels for the basketball
    object_points = initial_position
    object_labels = np.array([1], dtype=np.int32)

    # Process video
    video_segments = process_video(predictor, str(temp_folder), object_points, object_labels)

    # Create output frames with bounding boxes
    create_output_frames(video_segments, str(temp_folder), str(output_frames), frame_count)

    print(f"Output frames with bounding boxes saved in directory: {output_frames}")

if __name__ == "__main__":
    main()
