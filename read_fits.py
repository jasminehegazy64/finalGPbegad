# import numpy as np
# from astropy.io import fits

# def extract_star_positions(fits_file_path):
#     """Extract star positions (x, y) from a FITS file."""
#     with fits.open(fits_file_path) as hdul:
#         # Assuming the star positions are stored in the first binary table extension
#         star_data = hdul[1].data  # Update the index if necessary

#         # Extract x and y positions
#         x_positions = star_data['X']
#         y_positions = star_data['Y']

#         return x_positions, y_positions

# # Example usage
# if __name__ == "__main__":
#     fits_file_path = r"C:\Users\USER\Desktop\finalGPbegad\axy.fits"  # Replace with your FITS file path
#     x_positions, y_positions = extract_star_positions(fits_file_path)

#     # Print star positions
#     print("Star Positions (x, y):")
#     for x, y in zip(x_positions, y_positions):
#         print(f"({x}, {y})")



import numpy as np
from astropy.io import fits

def list_columns(fits_file_path):
    """List columns in the first binary table extension of a FITS file."""
    with fits.open(fits_file_path) as hdul:
        print(hdul[1].columns)

# Example usage
if __name__ == "__main__":
    fits_file_path = r"C:\Users\USER\Desktop\finalGPbegad\axy.fits"  # Replace with your FITS file path
    list_columns(fits_file_path)
