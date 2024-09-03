import os

import matplotlib.pyplot as plt
import numpy as np
from common import remove_background_box, remove_background_points
from PIL import Image


def find_input_image(filename='input.png'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tutorial_root = os.path.dirname(current_dir)
    image_path = os.path.join(tutorial_root, filename)
    if os.path.exists(image_path):
        return image_path
    image_path = os.path.join(current_dir, filename)
    if os.path.exists(image_path):
        return image_path
    raise FileNotFoundError(f"Could not find {filename} in the tutorial root or current directory.")

def plot_points_on_image(image, points, labels):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for point, label in zip(points, labels):
        color = 'r' if label == 0 else 'g'
        plt.plot(point[0], point[1], f'{color}o', markersize=10)
    plt.title("Input Image with Marked Points")
    plt.axis('off')
    return plt.gcf()

def plot_box_on_image(image, x1, y1, x2, y2):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                        fill=False, edgecolor='r', linewidth=2))
    plt.title("Input Image with Bounding Box")
    plt.axis('off')
    return plt.gcf()

def main():
    try:
        image_path = find_input_image()
        image = np.array(Image.open(image_path))
        print(f"Successfully loaded image from: {image_path}")
        
        # Get the tutorial folder path
        tutorial_folder = os.path.dirname(os.path.abspath(__file__))
        
        # Smart point selection: 4 inside, 4 outside
        points = [
            [200, 150],  # Inside kitten's body
            [350, 75],  # Inside kitten's head
            [450, 250],  # Inside kitten's paw
            [150, 220],  # Inside kitten's back paw
            [50, 50],    # Outside top-left
            [550, 50],   # Outside top-right
            [550, 300],  # Outside bottom-right
            [50, 300]    # Outside bottom-left
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 for foreground (kitten), 0 for background
        
        # Create and save the points visualization
        points_fig = plot_points_on_image(image, points, labels)
        points_fig.savefig(os.path.join(tutorial_folder, "input_image_with_points.png"))
        plt.close(points_fig)
        print("Points visualization saved as 'input_image_with_points.png' in the tutorial folder.")
        
        processed_image_points, comparison_points = remove_background_points(image, points, labels)
        processed_image_points.save(os.path.join(tutorial_folder, "processed_image_points.png"))
        comparison_points.save(os.path.join(tutorial_folder, "comparison_points.png"))
        print("Point-based processing complete. Check 'processed_image_points.png' and 'comparison_points.png' in the tutorial folder.")
        
        # Box-based segmentation
        x1, y1, x2, y2 = 50, 20, 550, 300  # Adjusted to fit better within the image
        
        # Create and save the box visualization
        box_fig = plot_box_on_image(image, x1, y1, x2, y2)
        box_fig.savefig(os.path.join(tutorial_folder, "input_image_with_box.png"))
        plt.close(box_fig)
        print("Box visualization saved as 'input_image_with_box.png' in the tutorial folder.")
        
        processed_image_box, comparison_box = remove_background_box(image, x1, y1, x2, y2)
        processed_image_box.save(os.path.join(tutorial_folder, "processed_image_box.png"))
        comparison_box.save(os.path.join(tutorial_folder, "comparison_box.png"))
        print("Box-based processing complete. Check 'processed_image_box.png' and 'comparison_box.png' in the tutorial folder.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'input.png' is in the tutorial root folder or the same folder as main.py.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()