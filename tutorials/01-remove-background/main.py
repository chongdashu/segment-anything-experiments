import os
import numpy as np
from PIL import Image
from common import remove_background_points, remove_background_box

def find_input_image(filename='input.png'):
    # Start from the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the tutorial root
    tutorial_root = os.path.dirname(current_dir)
    
    # Check if the file exists in the tutorial root
    image_path = os.path.join(tutorial_root, filename)
    if os.path.exists(image_path):
        return image_path
    
    # If not found, check in the current directory
    image_path = os.path.join(current_dir, filename)
    if os.path.exists(image_path):
        return image_path
    
    raise FileNotFoundError(f"Could not find {filename} in the tutorial root or current directory.")

def main():
    try:
        # Find and load the input image
        image_path = find_input_image()
        image = np.array(Image.open(image_path))
        print(f"Successfully loaded image from: {image_path}")
        
        # Example using points
        points = [[100, 100], [200, 200], [300, 300]]
        labels = [1, 1, 0]  # 1 for foreground, 0 for background
        processed_image_points, comparison_points = remove_background_points(image, points, labels)
        processed_image_points.save("processed_image_points.png")
        comparison_points.save("comparison_points.png")
        print("Point-based processing complete. Check 'processed_image_points.png' and 'comparison_points.png'.")
        
        # Example using bounding box
        x1, y1, x2, y2 = 100, 100, 400, 400
        processed_image_box, comparison_box = remove_background_box(image, x1, y1, x2, y2)
        processed_image_box.save("processed_image_box.png")
        comparison_box.save("comparison_box.png")
        print("Box-based processing complete. Check 'processed_image_box.png' and 'comparison_box.png'.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'input.png' is in the tutorial root folder or the same folder as main.py.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()