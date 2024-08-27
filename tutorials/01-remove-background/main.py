import os
import numpy as np
from PIL import Image
from common import remove_background_points, remove_background_box

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
            [350, 200],  # Inside kitten's head
            [450, 250],  # Inside kitten's paw
            [150, 300],  # Inside kitten's back paw
            [50, 50],    # Outside top-left
            [550, 50],   # Outside top-right
            [550, 350],  # Outside bottom-right
            [50, 350]    # Outside bottom-left
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 for foreground (kitten), 0 for background
        
        processed_image_points, comparison_points = remove_background_points(image, points, labels)
        processed_image_points.save(os.path.join(tutorial_folder, "processed_image_points.png"))
        comparison_points.save(os.path.join(tutorial_folder, "comparison_points.png"))
        print("Point-based processing complete. Check 'processed_image_points.png' and 'comparison_points.png' in the tutorial folder.")
        
        # Box-based segmentation (unchanged)
        x1, y1, x2, y2 = 50, 20, 580, 350
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