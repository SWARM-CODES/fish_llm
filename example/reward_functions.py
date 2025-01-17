import numpy as np

def reward_function_1(trajectories_x, trajectories_y, trajectories_z, bathymetry, time_window, target_locations, radius=5.0, depth_threshold=5.0):
    """
    Reward function for fish larvae settlement based on proximity to target locations and bathymetry.
    """
    start_step, end_step = time_window
    rewards = np.zeros(trajectories_x.shape[0])

    for i in range(rewards.size):
        for step in range(start_step, end_step + 1):
            x, y, z = trajectories_x[i, step], trajectories_y[i, step], trajectories_z[i, step]

            for target_x, target_y in target_locations:
                if np.sqrt((x - target_x)**2 + (y - target_y)**2) <= radius:
                    grid_x, grid_y = int(target_x / 10), int(target_y / 10)
                    expected_depth = bathymetry[grid_y, grid_x]
                    if abs(z - expected_depth) <= depth_threshold:
                        rewards[i] = 1
                        break
    return rewards


def reward_function_2(trajectories_x, trajectories_y, trajectories_z, temperature_field_with_noise, bathymetry, sigma_layers, time_window, temperature_range=(17, 20), depth_threshold=5.0, bathymetry_range=(-30,-20)):
    """
    Reward function for fish larvae settlement based on environmental conditions like temperature and bathymetry.
    """
    start_step, end_step = time_window
    rewards = np.zeros(trajectories_x.shape[0])

    for i in range(rewards.size):
        settled = False
        for step in range(start_step, end_step + 1):
            x, y, z = trajectories_x[i, step], trajectories_y[i, step], trajectories_z[i, step]

            grid_x_idx = (np.abs(np.arange(bathymetry.shape[0]) - x)).argmin()
            grid_y_idx = (np.abs(np.arange(bathymetry.shape[1]) - y)).argmin()

            expected_depth = bathymetry[grid_y_idx, grid_x_idx]
            if bathymetry_range[0] <= expected_depth <= bathymetry_range[1]:
                # Check if the particle is near the expected depth
                if abs(z - expected_depth) <= depth_threshold:
                    # Find the temperature at the particle's position
                    z_layer_idx = (np.abs(sigma_layers - z / expected_depth)).argmin()
                    temperature = temperature_field_with_noise[grid_x_idx, grid_y_idx, z_layer_idx]

                    # Check if the temperature is within the suitable range
                    if temperature_range[0] <= temperature <= temperature_range[1]:
                        settled = True
                        break  # Stop checking once settled


        rewards[i] = 1 if settled else 0
    return rewards

