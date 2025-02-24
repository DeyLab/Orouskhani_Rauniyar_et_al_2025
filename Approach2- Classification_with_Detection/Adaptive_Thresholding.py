import cv2
import numpy as np
import os
import glob

# Folder containing the PNG microscopy images
input_folder = "input_images"  # Change this to your folder path
output_base_dir = "cropped_cells"  # Base directory to store cropped cells
os.makedirs(output_base_dir, exist_ok=True)

# Define a list of C values to try with adaptive thresholding
c_values = [5, 7, 9]  # You can adjust or expand this list as needed

# Parameters for detecting cells of size ~50x50 pixels (with some tolerance)
target_width = 50
target_height = 50
tolerance = 5  # Allowable deviation in pixels

# Block size for adaptive thresholding (must be odd and > 1)
blockSize = 11

# Get all PNG images in the input folder
image_paths = glob.glob(os.path.join(input_folder, "*.png"))

for image_path in image_paths:
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        continue

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get a base name for the image (without extension)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Loop over each C value for adaptive thresholding
    for c in c_values:
        # Apply adaptive thresholding using the Gaussian method
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, blockSize, c)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an output subdirectory for the current C value and image
        output_dir = os.path.join(output_base_dir, f"C_{c}", base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        cell_count = 0
        # Loop over each contour detected
        for contour in contours:
            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the detected region is approximately 50x50 pixels
            if abs(w - target_width) <= tolerance and abs(h - target_height) <= tolerance:
                # Crop the region (cell) from the original image
                cell_crop = img[y:y+h, x:x+w]
                cell_count += 1
                # Save the cropped cell
                output_path = os.path.join(output_dir, f"cell_{cell_count}.png")
                cv2.imwrite(output_path, cell_crop)
        
        print(f"Image '{base_name}', C value {c}: Saved {cell_count} cropped cells in '{output_dir}'.")
