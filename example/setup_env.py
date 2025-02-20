import numpy as np
from netCDF4 import Dataset

# Define domain parameters
x_dim = 50   # 50 grid points in x-direction
y_dim = 50   # 50 grid points in y-direction
z_dim = 20   # 20 sigma layers
grid_resolution = 10  # 1 km resolution per grid cell
num_days = 30  # Number of days for simulation

# Define velocity and bathymetry parameters
surface_velocity = -0.5   # m/s, east-to-west flow at the surface
velocity_decrease = 0.02  # m/s, decrease in velocity per sigma layer to bottom
bathymetry_shallow = -10  # Depth at the western edge in meters
bathymetry_deep = -100    # Depth at the eastern edge in meters

def create_daily_velocity_fields(u_velocity_climatology, day, num_days, v_max_variation=0.5):
    """
    Generate daily velocity fields with climatology as the background.
    
    Parameters:
        u_velocity_climatology: 3D numpy array (x, y, z), background u_velocity field.
        day: int, current day of simulation.
        num_days: int, total number of days in the simulation.
        v_max_variation: float, maximum variation for v_velocity field (default: 0.5 m/s).
    
    Returns:
        daily_u_velocity: 3D numpy array (x, y, z) with daily variation for u_velocity.
        daily_v_velocity: 3D numpy array (x, y, z) with daily variation for v_velocity.
        daily_w_velocity: 3D numpy array (x, y, z), vertical velocity (remains zero).
    """
    # Add sinusoidal daily variation to u_velocity
    random_variation = 1 + 0.1 * np.sin(2 * np.pi * day / num_days)  # ±10% periodic variation
    daily_u_velocity = u_velocity_climatology * random_variation

    # Generate random daily variations for v_velocity (random in each grid point)
    daily_v_velocity = np.random.uniform(-v_max_variation, v_max_variation, u_velocity_climatology.shape)

    # Vertical velocity (w) remains zero
    daily_w_velocity = np.zeros_like(u_velocity_climatology)

    return daily_u_velocity, daily_v_velocity, daily_w_velocity
def calculate_signal_map(X, Y, reef_locations, threshold=15, decay_factor=-0.20):
    """
    Calculate the average signal strength at each grid point due to coral reef influence.

    Parameters:
        X, Y: 2D meshgrid of x, y coordinates.
        reef_locations: List of (x, y) coordinates of reef locations.
        threshold: Distance beyond which the signal rapidly decays (~15 km).
        decay_factor: Controls the rate of exponential decay.

    Returns:
        signal_map: 2D array with the averaged signal strength at each grid point.
    """
    signal_map = np.zeros_like(X, dtype=float)
    #weight_sum = np.zeros_like(X, dtype=float)  # To count contributing reefs at each grid point

    for reef_x, reef_y in reef_locations:
        # Compute Euclidean distance from each grid point to the reef location
        distances = np.sqrt((X - reef_x) ** 2 + (Y - reef_y) ** 2)

        # Apply diffusive decay function
        signal = np.exp(decay_factor * distances)

        #weights = np.exp(-distances)
        #weights[distances > threshold] = 0
        signal_map = np.maximum(signal_map, signal)
        # Sum up contributions and track the number of reefs affecting each point
       # signal_map += signal * weights
       # weight_sum += weights

    # Compute the average signal by dividing by the number of contributors
    #signal_map[weight_sum > 0] /= weight_sum[weight_sum > 0]

    return signal_map

def setup_environment():
    """
    Precompute and save environmental factors as a 4D matrix (t, x, y, z).
    """
    # Create grid coordinates
    x = np.arange(0, x_dim * grid_resolution, grid_resolution)  # x-coordinates
    y = np.arange(0, y_dim * grid_resolution, grid_resolution)  # y-coordinates
    sigma_layers = np.linspace(0, -1, z_dim)  # Sigma layers from surface (0) to bottom (-1)
    increase_rate = 2.0
    depth = bathymetry_shallow - np.arange(x_dim) * increase_rate
    # Generate bathymetry
    bathymetry = np.tile(depth, (y_dim, 1))
    z = np.array([bathymetry * sigma for sigma in sigma_layers])
    x_grid, y_grid, z_grid = np.meshgrid(x, y, sigma_layers, indexing='xy')
   
    reef_locations = np.loadtxt("reef_location.txt") 
    signal_map = calculate_signal_map(x_grid[:, :, 0], y_grid[:, :, 0], reef_locations)
    #print(signal_map.shape)  # Expected output: (50, 50)


    # Create initial u_velocity (base velocity without time variation)
    u_velocity_climatology = np.zeros((x_dim, y_dim, z_dim))
    for k in range(z_dim):
        u_velocity_climatology[:, :, k] = surface_velocity + k * velocity_decrease

    # Initialize temperature field
    surface_temp_min = 20  # Surface temperature at the west
    surface_temp_max = 30  # Surface temperature at the east
    bottom_temp_min = 10    # Bottom temperature at the east
    bottom_temp_max = 25   # Bottom temperature at the west
    surface_temps = np.linspace(surface_temp_max, surface_temp_min, x_dim)
    bottom_temps = np.linspace(bottom_temp_max, bottom_temp_min, x_dim)

    temperature_field = np.zeros((y_dim, x_dim, z_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            temperature_field[j, i, :] =  np.linspace(surface_temps[i], bottom_temps[i], z_dim)
    noise = np.random.uniform(-1, 1, (y_dim, x_dim, z_dim))
    temperature_field = temperature_field + noise
    #temperature_field[0:15, 0:15, :] -= 5  

    # Save environment data to a NetCDF file
    with Dataset("environment_data.nc", "w", format="NETCDF4") as nc:
        # Define dimensions
        nc.createDimension("time", num_days)
        nc.createDimension("x", x_dim)
        nc.createDimension("y", y_dim)
        nc.createDimension("z", z_dim)
        
        # Create variables for grid coordinates
        

        # Create variables
        u_var = nc.createVariable("u_velocity", "f4", ("time", "y", "x", "z"))
        v_var = nc.createVariable("v_velocity", "f4", ("time", "y", "x", "z"))
        w_var = nc.createVariable("w_velocity", "f4", ("time", "y", "x", "z"))
        temp_var = nc.createVariable("temperature", "f4", ("time", "y", "x", "z"))
        bathy_var = nc.createVariable("bathymetry", "f4", ("y", "x"))
        coral_var = nc.createVariable("coral_signal", "f4", ("y", "x"))
        x_grid_var = nc.createVariable("x_grid", "f4", ("y", "x", "z"))
        y_grid_var = nc.createVariable("y_grid", "f4", ("y", "x", "z"))
        z_grid_var = nc.createVariable("z_grid", "f4", ("y", "x", "z"))
    



        # Generate and save daily velocity fields
        for day in range(num_days):
            daily_u_velocity, daily_v_velocity, daily_w_velocity = create_daily_velocity_fields(
                u_velocity_climatology, day, num_days
            )
            u_var[day, :, :, :] = daily_u_velocity
            v_var[day, :, :, :] = daily_v_velocity
            w_var[day, :, :, :] = daily_w_velocity

            # Vary temperature slightly over time
            daily_temperature = temperature_field #* (1 + 0.01 * np.sin(2 * np.pi * day / num_days))
            temp_var[day, :, :, :] = daily_temperature

        # Save static fields
        bathy_var[:, :] = bathymetry
        coral_var[:, :] = signal_map
        x_grid_var[:, :, :] = x_grid
        y_grid_var[:, :, :] = y_grid
        z_grid_var[:, :, :] = z_grid

    print("Environment data saved to 'environment_data.nc'")

# Execute setup
if __name__ == "__main__":
    setup_environment()

