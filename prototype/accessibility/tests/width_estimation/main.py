import os
import csv
import re
from detection import ObstacleDetector
from estimation import WidthEstimator

def parse_filename(filename):
    """Extract ID and Heading from filenames"""
    try:
        id_part = re.search(r'_id(\d+)', filename).group(1)
        heading_part = re.search(r'_heading(\d+)', filename).group(1)
        return int(id_part), int(heading_part)
    except (AttributeError, ValueError) as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return None, None

def process_folder(input_folder, output_csv):
    detector = ObstacleDetector()
    estimator = WidthEstimator()
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1] in image_extensions
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Heading', 'Class', 'Width (m)', 'Margin (m)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for img_file in image_files:
            image_path = os.path.join(input_folder, img_file)
            img_id, heading = parse_filename(img_file)
            
            if img_id is None or heading is None:
                continue  # Skip invalid filenames
                
            try:
                detection_results = detector.detect_obstacles(image_path)
                final_results = estimator.estimate_widths(
                    image_path,
                    detection_results['segmentation_map'],
                    detection_results['obstacles']
                )
                
                for result in final_results:
                    writer.writerow({
                        'Id': img_id,
                        'Heading': heading,
                        'Class': result['class'],
                        'Width (m)': result['width_m'],
                        'Margin (m)': result['margin_m']
                    })
                
                print(f"Processed: {img_file} (ID: {img_id}, Heading: {heading})")
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")

if __name__ == "__main__":
    input_folder = "generic/images"
    output_csv = "obstacle_width_results_30.csv"
    process_folder(input_folder, output_csv)