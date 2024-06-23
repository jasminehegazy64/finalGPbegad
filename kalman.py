# # from astropy.io import fits
# # from astropy.wcs import WCS

# # # Load the FITS file
# # fits_file = r'C:\Users\USER\Desktop\finalGPbegad\fits\NEOS_SCI_2024001000555.fits'
# # hdulist = fits.open(fits_file)

# # # Get the WCS (World Coordinate System) information
# # w = WCS(hdulist[0].header)

# # # Print the WCS information
# # print(w)

# # # Optionally, you can get specific header values
# # ra = hdulist[0].header['RA']
# # dec = hdulist[0].header['DEC']

# # print(f"RA: {ra}, DEC: {dec}")

# # import numpy as np
# # from astropy.io import fits
# # from astropy.wcs import WCS
# # from photutils import DAOStarFinder
# # from astropy.stats import mad_std

# # # Load the FITS file
# # fits_file = r'C:\Users\USER\Desktop\finalGPbegad\fits\NEOS_SCI_2024001000555.fits'
# # hdulist = fits.open(fits_file)
# # data = hdulist[0].data

# # # Get WCS information
# # w = WCS(hdulist[0].header)

# # # Detect stars in the image
# # mean, median, std = np.mean(data), np.median(data), mad_std(data)
# # daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
# # sources = daofind(data - median)

# # # Convert pixel coordinates to celestial coordinates
# # for source in sources:
# #     x = source['xcentroid']
# #     y = source['ycentroid']
# #     ra, dec = w.wcs_pix2world(x, y, 1)
# #     print(f"Object at x={x}, y={y} has RA={ra}, DEC={dec}")


# # import numpy as np

# # # Assume you have a list of (time, ra, dec) tuples
# # observations = [
# #     (time1, ra1, dec1),
# #     (time2, ra2, dec2),
# #     # add more observations
# # ]

# # velocities = []
# # for i in range(1, len(observations)):
# #     t1, ra1, dec1 = observations[i-1]
# #     t2, ra2, dec2 = observations[i]
# #     dt = t2 - t1  # time difference in seconds or appropriate unit
    
# #     # Calculate angular distance
# #     delta_ra = (ra2 - ra1) * np.cos(np.radians((dec1 + dec2) / 2))
# #     delta_dec = dec2 - dec1
# #     angular_distance = np.sqrt(delta_ra**2 + delta_dec**2)  # in degrees
    
# #     # Convert to arcseconds if needed
# #     angular_distance *= 3600
    
# #     # Calculate velocity (arcseconds per time unit)
# #     velocity = angular_distance / dt
# #     velocities.append(velocity)

# # # Print velocities
# # for v in velocities:
# #     print(f"Velocity: {v} arcseconds per unit time")





# from astropy.io import fits
# from astropy.wcs import WCS
# from photutils import DAOStarFinder
# from astropy.stats import mad_std
# import numpy as np

# # Load the FITS file
# fits_file = r'C:\Users\USER\Desktop\finalGPbegad\fits\NEOS_SCI_2024001000555.fits'
# hdulist = fits.open(fits_file)
# data = hdulist[0].data

# # Get WCS information
# w = WCS(hdulist[0].header)

# # Detect stars in the image
# mean, median, std = np.mean(data), np.median(data), mad_std(data)
# daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
# sources = daofind(data - median)

# # Get observation time from the FITS header
# jd_obs = hdulist[0].header['JD-OBS']

# # Print the JD observation time
# print(f"Observation Time (JD): {jd_obs}")

# # Convert pixel coordinates to celestial coordinates
# observations = []
# for source in sources:
#     x = source['xcentroid']
#     y = source['ycentroid']
#     ra, dec = w.wcs_pix2world(x, y, 1)
#     observations.append((jd_obs, ra, dec))

# # Print the observations array
# print("Observations array:")
# for obs in observations:
#     print(obs)

# # Calculate velocities between consecutive observations
# velocities = []
# for i in range(1, len(observations)):
#     t1, ra1, dec1 = observations[i-1]
#     t2, ra2, dec2 = observations[i]
#     dt = t2 - t1  # Time difference in Julian Days

#     # Calculate angular distance
#     delta_ra = (ra2 - ra1) * np.cos(np.radians((dec1 + dec2) / 2))
#     delta_dec = dec2 - dec1
#     angular_distance = np.sqrt(delta_ra**2 + delta_dec**2)  # in degrees

#     # Convert to arcseconds if needed
#     angular_distance *= 3600

#     # Calculate velocity (arcseconds per time unit)
#     velocity = angular_distance / dt
#     velocities.append(velocity)

# # Print velocities
# print("Velocities:")
# for v in velocities:
#     print(f"Velocity: {v} arcseconds per unit time")




import numpy as np
import plotly.graph_objs as go
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy.stats import mad_std
from astropy.time import Time

# Load the FITS file
fits_file = r'C:\Users\USER\Desktop\finalGPbegad\fits\NEOS_SCI_2024001000555.fits'
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

# Calculate velocities between consecutive observations
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

# Define Earth as a 3D sphere
def plot_earth():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Convert RA/Dec to Cartesian coordinates
def spherical_to_cartesian(ra, dec, r=6371 + 400):  # Assuming objects are 400 km above Earth surface
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return x, y, z

# Get Earth data
x_earth, y_earth, z_earth = plot_earth()

# Create scatter plot for object positions
x_positions = []
y_positions = []
z_positions = []

for _, ra, dec, _ in velocities:
    x, y, z = spherical_to_cartesian(ra, dec)
    x_positions.append(x)
    y_positions.append(y)
    z_positions.append(z)

# Create 3D plot
fig = go.Figure()

# Add Earth
fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.6))

# Add object positions
fig.add_trace(go.Scatter3d(
    x=x_positions, y=y_positions, z=z_positions,
    mode='markers',
    marker=dict(size=4, color='red', symbol='circle')
))

# Set plot layout
fig.update_layout(
    title='Object Motion Around the Earth',
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data'
    )
)

fig.show()


