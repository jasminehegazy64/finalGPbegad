import os
import numpy as np
import plotly.graph_objs as go
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy.stats import mad_std
from astropy.time import Time
from pykalman import KalmanFilter

def load_fits_files(folder_path):
    fits_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.fits')]
    return fits_files

def process_fits_file(fits_file):
    hdulist = fits.open(fits_file)
    data = hdulist[0].data

    # Get WCS information
    w = WCS(hdulist[0].header)

    # Detect stars in the image
    mean, median, std = np.mean(data), np.median(data), mad_std(data)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
    sources = daofind(data - median)

    # Get observation time from the FITS header
    date_obs = hdulist[0].header['DATE-OBS']
    jd_obs = Time(date_obs, format='isot', scale='utc').jd

    # Convert pixel coordinates to celestial coordinates
    observations = []
    for source in sources:
        x = source['xcentroid']
        y = source['ycentroid']
        ra, dec = w.wcs_pix2world(x, y, 1)
        observations.append((jd_obs, ra, dec))
    
    return observations

def calculate_velocities(observations):
    velocities = []
    for i in range(1, len(observations)):
        t1, ra1, dec1 = observations[i-1]
        t2, ra2, dec2 = observations[i]
        dt = (t2 - t1) * 86400  # Time difference in seconds

        # Calculate angular distance
        delta_ra = (ra2 - ra1) * np.cos(np.radians((dec1 + dec2) / 2))
        delta_dec = dec2 - dec1
        angular_distance = np.sqrt(delta_ra**2 + delta_dec**2)  # in degrees

        # Convert to arcseconds
        angular_distance *= 3600

        # Calculate velocity (arcseconds per second)
        velocity = angular_distance / dt
        velocities.append((t2, ra2, dec2, velocity))
    
    return velocities

def plot_earth():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def spherical_to_cartesian(ra, dec, r=6371 + 400):  # Assuming objects are 400 km above Earth surface
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return x, y, z

def apply_kalman_filter(observations):
    initial_state = np.array([observations[0][1], observations[0][2], 0, 0])  # initial state (ra, dec, ra_velocity, dec_velocity)
    initial_state_covariance = np.eye(4) * 1e-4  # initial state covariance

    transition_matrix = np.eye(4)
    observation_matrix = np.eye(4)[:2, :4]

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state,
        initial_state_covariance=initial_state_covariance,
    )

    # Extract RA and DEC observations
    observations_array = np.array([(obs[1], obs[2]) for obs in observations])

    # Fit the filter to the data
    kf = kf.em(observations_array, n_iter=5)

    # Apply the filter
    states, state_covariances = kf.filter(observations_array)

    return states, state_covariances

def main(folder_path):
    fits_files = load_fits_files(folder_path)

    all_observations = []
    for fits_file in fits_files:
        observations = process_fits_file(fits_file)
        all_observations.extend(observations)
    
    all_observations.sort()  # Sort by observation time

    states, state_covariances = apply_kalman_filter(all_observations)

    # Get Earth data
    x_earth, y_earth, z_earth = plot_earth()

    # Create scatter plot for object positions
    x_positions = []
    y_positions = []
    z_positions = []
    x_predicted = []
    y_predicted = []
    z_predicted = []

    for i, (jd, ra, dec) in enumerate(all_observations):
        x, y, z = spherical_to_cartesian(ra, dec)
        x_positions.append(x)
        y_positions.append(y)
        z_positions.append(z)

    for state in states:
        ra_pred, dec_pred = state[0], state[1]
        x_pred, y_pred, z_pred = spherical_to_cartesian(ra_pred, dec_pred)
        x_predicted.append(x_pred)
        y_predicted.append(y_pred)
        z_predicted.append(z_pred)

    # Create 3D plot
    fig = go.Figure()

    # Add Earth
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.6))

    # Add object positions
    fig.add_trace(go.Scatter3d(
        x=x_positions, y=y_positions, z=z_positions,
        mode='markers',
        marker=dict(size=4, color='red', symbol='circle'),
        name='Observed Positions'
    ))

    # Add predicted positions
    fig.add_trace(go.Scatter3d(
        x=x_predicted, y=y_predicted, z=z_predicted,
        mode='lines',
        line=dict(color='green', width=2),
        name='Predicted Positions'
    ))

    # Set plot layout
    fig.update_layout(
        title='Object Motion Around the Earth with Kalman Filter Prediction',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        )
    )

    fig.show()

# Example usage
if __name__ == "__main__":
    folder_path = r'C:\Users\USER\Desktop\finalGPbegad\fits'
    main(folder_path)
