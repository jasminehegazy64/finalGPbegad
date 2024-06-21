import csv
import numpy as np
from astropy.io import fits

import csv

def read_star_coordinates_from_csv(csv_file_path, prediction_column_index,imagename_index, image_name='NEOS_SCI_2024001000555.png' ,star_label='Celestial Object'):
    """Read (x, y) coordinates from a CSV file and store them in a list if they are predicted to be stars."""
    coordinates = []
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        
        for row in csvreader:
            prediction = row[prediction_column_index]  # Read the prediction column
            name=row[imagename_index]
            if prediction == star_label and name==image_name:
                x = float(row[4])  # Assuming x is in the first column
                y = float(row[5])  # Assuming y is in the second column
                coordinates.append((x, y))
    
    return coordinates

# Example usage
csv_file_path = r'C:\Users\USER\Desktop\finalGPbegad\sim_debris.csv'  # Replace with your actual path
prediction_column_index = 10  # Replace with the actual index of the prediction column
image_index=0
coordinates = read_star_coordinates_from_csv(csv_file_path, prediction_column_index,image_index)
print(coordinates)




def extract_detected_stars(fits_file_path):
    """Extract detected star positions (x, y) from the FITS file."""
    with fits.open(fits_file_path) as hdul:
        detected_stars_data = hdul[1].data  # Assuming stars detected data is in the first extension
        star_x = detected_stars_data['X']  # Replace with the actual column name
        star_y = detected_stars_data['Y']  # Replace with the actual column name
        return np.array(star_x), np.array(star_y)

# Example usage
fits_file_path = r'C:\Users\USER\Desktop\finalGPbegad\axy.fits'  # Replace with your actual path
star_x, star_y = extract_detected_stars(fits_file_path)


from scipy.spatial import distance

def match_detections(your_detections, astrometry_detections, threshold=5):
    """Match your detections with astrometry.net detections based on a distance threshold."""
    matched = []
    for y_x, y_y in your_detections:
        for a_x, a_y in zip(astrometry_detections[0], astrometry_detections[1]):
            if distance.euclidean((y_x, y_y), (a_x, a_y)) < threshold:
                matched.append((y_x, y_y, a_x, a_y))
                break
    return matched

matched_detections = match_detections(coordinates, (star_x, star_y))

def calculate_metrics(your_detections, matched_detections, total_astrometry_detections):
    tp = len(matched_detections)
    fp = len(your_detections) - tp
    fn = total_astrometry_detections - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Example usage
total_astrometry_detections = len(star_x)  # Total stars detected by astrometry.net
precision, recall, f1_score = calculate_metrics(coordinates, matched_detections, total_astrometry_detections)
print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")
