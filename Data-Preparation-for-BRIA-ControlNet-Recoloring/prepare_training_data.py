
"""
Image Processing and Color Extraction Script
------------------------------------------

This script processes a folder of images to:
1. Extract dominant colors using K-means clustering
2. Save the color information to text files
3. Create grayscale versions of the images

The script consists of two main classes:
- ColorExtractor: Handles the color extraction using K-means clustering
- ImageProcessor: Manages the overall processing workflow

The process for each image is:
1. Read the image using OpenCV
2. Extract dominant colors using K-means clustering
3. Convert colors from BGR to RGB format
4. Save color information to a text file
5. Create and save a grayscale version

Required packages:
- numpy: For numerical operations
- opencv-python (cv2): For image processing
- pillow (PIL): For image operations
- scikit-learn: For K-means clustering
"""

import os
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ColorExtractor:
    """
    Extracts dominant colors from images using K-means clustering.
    This class reduces the image's color space to a specified number
    of representative colors.
    """
    
    def __init__(self, n_colors: int = 8, min_cluster_size: float = 0.005):
        """
        Initialize the color extractor with clustering parameters
        
        Args:
            n_colors: Number of colors to extract (K in K-means)
            min_cluster_size: Minimum size of a color cluster as a fraction
                            of total pixels (filters out rare colors)
        """
        self.n_colors = n_colors
        self.min_cluster_size = min_cluster_size
    
    def extract_colors(self, image_path: str) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from an image using K-means clustering.
        The process:
        1. Read the image
        2. Reshape to a list of pixels
        3. Perform K-means clustering
        4. Filter clusters by size
        5. Return significant colors
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of RGB color tuples, sorted by brightness
        """
        try:
            # Read image in BGR format (OpenCV default)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Reshape to 2D array of pixels
            pixels = image.reshape(-1, 3)
            
            # Perform k-means clustering to find dominant colors
            kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers (colors) and labels (pixel assignments)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count pixels in each cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_pixels = len(pixels)
            
            # Filter out colors that don't meet minimum cluster size
            significant_colors = []
            for label, count in zip(unique_labels, counts):
                if count / total_pixels >= self.min_cluster_size:
                    color = tuple(map(int, colors[label]))
                    significant_colors.append(color)
            
            # Sort colors by brightness (sum of RGB values)
            return sorted(significant_colors, key=lambda x: sum(x), reverse=True)
            
        except Exception as e:
            print(f"Error in color extraction: {str(e)}")
            return []

class ImageProcessor:
    """
    Main class for processing images. Handles:
    - Directory management
    - Image processing workflow
    - File saving operations
    """
    
    def __init__(self, input_folder: str, output_folder: str):
        """
        Initialize the image processor with input/output paths
        
        Args:
            input_folder: Path to folder containing input images
            output_folder: Path to output folder where results will be saved
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.condition_folder = os.path.join(output_folder, "condition")
        self.captions_folder = os.path.join(output_folder, "captions")
        
        # Create all necessary output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.condition_folder, exist_ok=True)
        os.makedirs(self.captions_folder, exist_ok=True)
        
        # Initialize color extractor with default parameters
        self.color_extractor = ColorExtractor(n_colors=8, min_cluster_size=0.005)
    
    def process_single_image(self, image_path: str) -> List[Tuple[int, int, int]]:
        """
        Process a single image to extract its dominant colors
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of RGB color tuples representing dominant colors
        """
        # Extract colors (in BGR format)
        bgr_colors = self.color_extractor.extract_colors(image_path)
        
        # Convert BGR to RGB format
        rgb_colors = [(c[2], c[1], c[0]) for c in bgr_colors]
        
        return rgb_colors
    
    def save_color_info(self, image_path: str, colors: List[Tuple[int, int, int]]):
        """
        Save color information to a text file and print to console
        
        Args:
            image_path: Path to source image
            colors: List of RGB color tuples to save
        """
        # Create readable color description
        color_text = "The colors of the image are: " + ", ".join([f"RGB{color}" for color in colors])
        
        # Generate output path for color information
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_path = os.path.join(self.captions_folder, f"{base_name}_colors.txt")
        
        # Save to file and print to console
        with open(caption_path, 'w') as f:
            f.write(color_text)
            
        print(f"Colors found: {color_text}")
        print(f"Saved color caption to: {caption_path}")
    
    def save_grayscale_condition(self, image_path: str):
        """
        Create and save a grayscale version of the input image
        
        Args:
            image_path: Path to source image
        """
        try:
            # Open image and convert to grayscale
            image = Image.open(image_path)
            grayscale = image.convert('L')
            
            # Generate output path and save
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.condition_folder, f"{base_name}_grayscale.jpg")
            grayscale.save(output_path)
            
            print(f"Saved grayscale image to: {output_path}")
            
        except Exception as e:
            print(f"Error saving grayscale for {image_path}: {str(e)}")
    
    def process_images(self):
        """
        Main processing function that handles all images in the input folder.
        For each image:
        1. Extracts colors
        2. Saves color information
        3. Creates grayscale version
        """
        # Get list of all image files
        image_files = [f for f in os.listdir(self.input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            print(f"\nProcessing: {image_path}")
            
            # Extract colors
            colors = self.process_single_image(image_path)
            
            if colors:
                # Save color information
                self.save_color_info(image_path, colors)
                
                # Create and save grayscale version
                self.save_grayscale_condition(image_path)
            else:
                print(f"Skipping {image_file} due to color extraction failure")

def main():
    """
    Main entry point of the script.
    Sets up paths and initializes the image processor.
    """
    # Define input and output paths
    input_folder = "data/images"
    output_folder = "data/"
    
    # Create and run processor
    processor = ImageProcessor(input_folder, output_folder)
    processor.process_images()

if __name__ == "__main__":
    main()
    
